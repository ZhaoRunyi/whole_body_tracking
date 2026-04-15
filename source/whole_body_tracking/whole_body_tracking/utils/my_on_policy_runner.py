import os
import statistics
import time

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import torch
import wandb
from whole_body_tracking.utils.ewc_regularizer import EwcConfig, PolicyEwcRegularizer
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: list[str] | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_names = list(registry_name) if registry_name is not None else []
        self.refpose_print_num_envs = 0
        self._motion_source_labels: list[str] | None = None
        self.ewc_regularizer: PolicyEwcRegularizer | None = None
        self._ewc_update_wrapped = False

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_names:
                for registry_name in self.registry_names:
                    wandb.run.use_artifact(registry_name)
                self.registry_names = []

    def configure_refpose_logging(self, num_envs: int) -> None:
        self.refpose_print_num_envs = max(int(num_envs), 0)

    def enable_ewc(
        self,
        *,
        ewc_lambda: float,
        ewc_fisher_batches: int,
        ewc_actor_only: bool,
    ) -> None:
        if not hasattr(self.env, "consume_ewc_replay_mask_rollout"):
            raise AttributeError(
                "EWC requires an env wrapper exposing consume_ewc_replay_mask_rollout(). "
                "Use EwcReplayMaskRslRlVecEnvWrapper when EWC is enabled."
            )

        self.ewc_regularizer = PolicyEwcRegularizer(
            EwcConfig(
                enable=True,
                lambda_=ewc_lambda,
                fisher_batches=ewc_fisher_batches,
                actor_only=ewc_actor_only,
            )
        )
        self._wrap_alg_update_for_ewc()

    def capture_ewc_reference_from_current_policy(self) -> None:
        if self.ewc_regularizer is None:
            return
        self.ewc_regularizer.capture_reference(self.alg.policy)

    def _wrap_alg_update_for_ewc(self) -> None:
        if self._ewc_update_wrapped:
            return

        original_update = self.alg.update

        def wrapped_update(*args, **kwargs):
            if self.ewc_regularizer is not None and not self.ewc_regularizer.fisher_ready:
                replay_mask_rollout = self.env.consume_ewc_replay_mask_rollout()
                self.ewc_regularizer.estimate_fisher_from_rollout(self.alg, replay_mask_rollout)
            elif hasattr(self.env, "consume_ewc_replay_mask_rollout"):
                self.env.consume_ewc_replay_mask_rollout()

            update_result = original_update(*args, **kwargs)

            if self.ewc_regularizer is not None:
                self.ewc_regularizer.apply_penalty_step(self.alg)

            return update_result

        self.alg.update = wrapped_update
        self._ewc_update_wrapped = True

    def _get_motion_command(self):
        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return None
        try:
            return command_manager.get_term("motion")
        except Exception:
            return None

    def _get_motion_source_labels(self, motion_files: list[str]) -> list[str]:
        if self._motion_source_labels is not None and len(self._motion_source_labels) == len(motion_files):
            return self._motion_source_labels

        basenames = [os.path.basename(path) for path in motion_files]
        if len(set(basenames)) == len(basenames):
            self._motion_source_labels = basenames
            return self._motion_source_labels

        try:
            common_root = os.path.commonpath(motion_files)
        except ValueError:
            common_root = ""
        if common_root:
            relpaths = [os.path.relpath(path, common_root) for path in motion_files]
            if len(set(relpaths)) == len(relpaths):
                self._motion_source_labels = relpaths
                return self._motion_source_labels

        self._motion_source_labels = list(motion_files)
        return self._motion_source_labels

    def _build_refpose_summary(self, pad: int) -> str:
        if self.refpose_print_num_envs <= 0:
            return ""

        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return ""

        max_envs = min(self.refpose_print_num_envs, motion_cmd.num_envs)
        if max_envs <= 0:
            return ""

        motion_source_labels = self._get_motion_source_labels(list(motion_cmd.motion_files))
        summary = ""
        for env_id in range(max_envs):
            motion_id = int(motion_cmd.motion_ids[env_id].item())
            time_step = int(motion_cmd.time_steps[env_id].item())
            motion_source = motion_source_labels[motion_id]
            summary += (
                f"""{f"RefPose/env[{env_id}]:":>{pad}} motion[{motion_id}] step={time_step} file={motion_source}\n"""
            )
        return summary

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs * getattr(self, "gpu_world_size", 1)
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        if self.ewc_regularizer is not None and self.ewc_regularizer.reference_ready:
            self.writer.add_scalar("Loss/ewc_penalty", self.ewc_regularizer.last_penalty, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            if self.ewc_regularizer is not None and self.ewc_regularizer.reference_ready:
                log_string += f"""{'EWC penalty:':>{pad}} {self.ewc_regularizer.last_penalty:.4f}\n"""
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
            if self.ewc_regularizer is not None and self.ewc_regularizer.reference_ready:
                log_string += f"""{'EWC penalty:':>{pad}} {self.ewc_regularizer.last_penalty:.4f}\n"""

        log_string += ep_string
        log_string += self._build_refpose_summary(pad)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)
