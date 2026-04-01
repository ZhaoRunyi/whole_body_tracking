from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


SAMPLING_PRESET_DEFAULTS = {
    "beyondmimic": {
        "motion_resample_scope": "episode_reset_and_rollover",
        "phase_sampling_window": "full_motion",
        "phase_sampling_strategy": "adaptive_legacy",
        "motion_end_behavior": "rollover_resample",
    },
    "hover": {
        "motion_resample_scope": "episode_reset_only",
        "phase_sampling_window": "truncate_to_episode",
        "phase_sampling_strategy": "uniform",
        "motion_end_behavior": "terminate_episode",
    },
    "hover_adaptive": {
        "motion_resample_scope": "episode_reset_only",
        "phase_sampling_window": "truncate_to_episode",
        "phase_sampling_strategy": "adaptive_per_motion",
        "motion_end_behavior": "terminate_episode",
    },
}


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = torch.device(device)
        body_indexes_np = np.asarray(body_indexes, dtype=np.int64)
        with np.load(motion_file, allow_pickle=False) as data:
            self.fps = data["fps"]
            self.joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
            self.joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
            # Only keep bodies required by the task to avoid wasting memory on unused references.
            self.body_pos_w = torch.as_tensor(
                data["body_pos_w"][:, body_indexes_np], dtype=torch.float32, device=self.device
            )
            self.body_quat_w = torch.as_tensor(
                data["body_quat_w"][:, body_indexes_np], dtype=torch.float32, device=self.device
            )
            self.body_lin_vel_w = torch.as_tensor(
                data["body_lin_vel_w"][:, body_indexes_np], dtype=torch.float32, device=self.device
            )
            self.body_ang_vel_w = torch.as_tensor(
                data["body_ang_vel_w"][:, body_indexes_np], dtype=torch.float32, device=self.device
            )
        self.time_step_total = self.joint_pos.shape[0]

    def sample(self, field_name: str, time_steps: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        field = getattr(self, field_name)
        sample = field[time_steps.to(device=field.device, dtype=torch.long)]
        target_device = torch.device(device)
        if sample.device != target_device:
            sample = sample.to(target_device)
        return sample


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        motion_files = self.cfg.motion_file
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        if len(motion_files) == 0:
            raise ValueError("motion_file cannot be empty.")
        self.motion_files = [os.path.abspath(path) for path in motion_files]

        motion_storage_device = self.cfg.motion_storage_device
        if motion_storage_device is None:
            motion_storage_device = "cpu" if len(motion_files) > 1 else self.device

        self.motions = [
            MotionLoader(path, self.body_indexes.detach().cpu().tolist(), device=motion_storage_device)
            for path in self.motion_files
        ]
        # Keep compatibility for legacy code paths (e.g. exporter).
        self.motion = self.motions[0]
        self.num_motions = len(self.motions)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.tensor([m.time_step_total for m in self.motions], dtype=torch.long, device=self.device)
        self._refpose_log_events: list[dict[str, int | str]] = []
        self._resample_reason = "unknown"

        if self.cfg.motion_sampling not in ("uniform", "weighted"):
            raise ValueError(f"Unsupported motion_sampling: {self.cfg.motion_sampling}")
        if self.cfg.motion_sampling == "weighted":
            if self.cfg.motion_weights is None or len(self.cfg.motion_weights) != self.num_motions:
                raise ValueError("motion_weights must match number of motion files when using weighted sampling.")
            prob = torch.tensor(self.cfg.motion_weights, dtype=torch.float32, device=self.device)
            self.motion_prob = prob / prob.sum()
        else:
            self.motion_prob = torch.ones(self.num_motions, dtype=torch.float32, device=self.device) / float(
                self.num_motions
            )
        self._resolve_sampling_policy(env)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_pos = torch.empty(
            (self.num_envs, *self.motion.joint_pos.shape[1:]), dtype=self.motion.joint_pos.dtype, device=self.device
        )
        self._joint_vel = torch.empty(
            (self.num_envs, *self.motion.joint_vel.shape[1:]), dtype=self.motion.joint_vel.dtype, device=self.device
        )
        self._body_pos_w = torch.empty(
            (self.num_envs, *self.motion.body_pos_w.shape[1:]), dtype=self.motion.body_pos_w.dtype, device=self.device
        )
        self._body_quat_w = torch.empty(
            (self.num_envs, *self.motion.body_quat_w.shape[1:]), dtype=self.motion.body_quat_w.dtype, device=self.device
        )
        self._body_lin_vel_w = torch.empty(
            (self.num_envs, *self.motion.body_lin_vel_w.shape[1:]),
            dtype=self.motion.body_lin_vel_w.dtype,
            device=self.device,
        )
        self._body_ang_vel_w = torch.empty(
            (self.num_envs, *self.motion.body_ang_vel_w.shape[1:]),
            dtype=self.motion.body_ang_vel_w.dtype,
            device=self.device,
        )
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self._refresh_motion_buffers()

        self.bin_count = int(self.motion_bin_counts[0].item())
        self.bin_failed_count = torch.zeros((self.num_motions, self.max_bin_count), dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros((self.num_motions, self.max_bin_count), dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    def _resolve_sampling_policy(self, env: ManagerBasedRLEnv) -> None:
        if self.cfg.sampling_preset not in SAMPLING_PRESET_DEFAULTS:
            raise ValueError(f"Unsupported sampling_preset: {self.cfg.sampling_preset}")

        defaults = SAMPLING_PRESET_DEFAULTS[self.cfg.sampling_preset]
        self.motion_resample_scope = self.cfg.motion_resample_scope or defaults["motion_resample_scope"]
        self.phase_sampling_window = self.cfg.phase_sampling_window or defaults["phase_sampling_window"]
        self.phase_sampling_strategy = self.cfg.phase_sampling_strategy or defaults["phase_sampling_strategy"]
        self.motion_end_behavior = self.cfg.motion_end_behavior or defaults["motion_end_behavior"]

        valid_motion_resample_scope = {"episode_reset_only", "episode_reset_and_rollover"}
        valid_phase_sampling_window = {"full_motion", "truncate_to_episode"}
        valid_phase_sampling_strategy = {"uniform", "adaptive_legacy", "adaptive_per_motion"}
        valid_motion_end_behavior = {"rollover_resample", "terminate_episode"}

        if self.motion_resample_scope not in valid_motion_resample_scope:
            raise ValueError(f"Unsupported motion_resample_scope: {self.motion_resample_scope}")
        if self.phase_sampling_window not in valid_phase_sampling_window:
            raise ValueError(f"Unsupported phase_sampling_window: {self.phase_sampling_window}")
        if self.phase_sampling_strategy not in valid_phase_sampling_strategy:
            raise ValueError(f"Unsupported phase_sampling_strategy: {self.phase_sampling_strategy}")
        if self.motion_end_behavior not in valid_motion_end_behavior:
            raise ValueError(f"Unsupported motion_end_behavior: {self.motion_end_behavior}")

        self.env_step_dt = env.cfg.decimation * env.cfg.sim.dt
        self.steps_per_bin = max(int(round(1.0 / self.env_step_dt)), 1)
        self.episode_length_steps = max(int(round(env.cfg.episode_length_s / self.env_step_dt)), 1)
        if self.phase_sampling_window == "truncate_to_episode":
            self.motion_sampling_max_starts = torch.clamp(self.motion_lengths - self.episode_length_steps, min=0)
        else:
            self.motion_sampling_max_starts = torch.clamp(self.motion_lengths - 1, min=0)

        motion_bin_counts: list[int] = []
        for motion_id in range(self.num_motions):
            if self.phase_sampling_strategy == "adaptive_legacy" and self.num_motions == 1:
                bin_count = int(self.motion.time_step_total // self.steps_per_bin) + 1
            else:
                sample_span = int(self.motion_sampling_max_starts[motion_id].item()) + 1
                bin_count = max(int(sample_span // self.steps_per_bin) + 1, 1)
            motion_bin_counts.append(bin_count)
        self.motion_bin_counts = torch.tensor(motion_bin_counts, dtype=torch.long, device=self.device)
        self.max_bin_count = int(self.motion_bin_counts.max().item())

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._resample_reason = "episode_reset"
        try:
            return super().reset(env_ids=env_ids)
        finally:
            self._resample_reason = "unknown"

    def consume_refpose_log_events(self, max_envs: int) -> list[dict[str, int | str]]:
        if max_envs <= 0:
            self._refpose_log_events.clear()
            return []

        events = [event for event in self._refpose_log_events if int(event["env_id"]) < max_envs]
        self._refpose_log_events.clear()
        return events

    def _refresh_motion_buffers(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if env_ids_tensor.numel() == 0:
                return

        for motion_id, motion in enumerate(self.motions):
            motion_env_ids = env_ids_tensor[self.motion_ids[env_ids_tensor] == motion_id]
            if motion_env_ids.numel() == 0:
                continue
            time_steps = torch.clamp(self.time_steps[motion_env_ids], max=motion.time_step_total - 1)
            self._joint_pos[motion_env_ids] = motion.sample("joint_pos", time_steps, self.device)
            self._joint_vel[motion_env_ids] = motion.sample("joint_vel", time_steps, self.device)
            self._body_pos_w[motion_env_ids] = motion.sample("body_pos_w", time_steps, self.device)
            self._body_quat_w[motion_env_ids] = motion.sample("body_quat_w", time_steps, self.device)
            self._body_lin_vel_w[motion_env_ids] = motion.sample("body_lin_vel_w", time_steps, self.device)
            self._body_ang_vel_w[motion_env_ids] = motion.sample("body_ang_vel_w", time_steps, self.device)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _set_uniform_sampling_metrics(self, env_ids: torch.Tensor) -> None:
        self.metrics["sampling_entropy"][env_ids] = 1.0
        self.metrics["sampling_top1_prob"][env_ids] = 0.0
        self.metrics["sampling_top1_bin"][env_ids] = 0.0

    def _sample_uniform_full_motion_legacy(self, env_ids: torch.Tensor) -> None:
        max_steps = self.motion_lengths[self.motion_ids[env_ids]]
        random_uniform = torch.rand(len(env_ids), dtype=torch.float32, device=self.device)
        self.time_steps[env_ids] = (random_uniform * (max_steps.float() - 1.0)).long()

    def _sample_uniform_time_steps(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        if self.phase_sampling_window == "full_motion" and self.phase_sampling_strategy == "adaptive_legacy":
            self._sample_uniform_full_motion_legacy(env_ids)
            self._set_uniform_sampling_metrics(env_ids)
            return

        sample_counts = self.motion_sampling_max_starts[self.motion_ids[env_ids]] + 1
        random_uniform = torch.rand(len(env_ids), dtype=torch.float32, device=self.device)
        self.time_steps[env_ids] = torch.floor(random_uniform * sample_counts.float()).long()
        self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], self.motion_sampling_max_starts[self.motion_ids[env_ids]])
        self._set_uniform_sampling_metrics(env_ids)

    def _update_failure_statistics_legacy(self, env_ids: torch.Tensor) -> None:
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if self.motion_end_behavior == "terminate_episode":
            episode_failed = episode_failed & (~self.motion_ended[env_ids])
        self._current_bin_failed[0].zero_()
        if not torch.any(episode_failed):
            return

        current_bin_index = torch.clamp(
            (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
        )
        fail_bins = current_bin_index[env_ids][episode_failed]
        self._current_bin_failed[0, : self.bin_count] = torch.bincount(fail_bins, minlength=self.bin_count)

    def _update_failure_statistics_per_motion(self, env_ids: torch.Tensor) -> None:
        self._current_bin_failed.zero_()
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if self.motion_end_behavior == "terminate_episode":
            episode_failed = episode_failed & (~self.motion_ended[env_ids])
        if not torch.any(episode_failed):
            return

        failed_env_ids = env_ids[episode_failed]
        failed_motion_ids = self.motion_ids[failed_env_ids]
        failed_time_steps = self.time_steps[failed_env_ids]
        for motion_id in torch.unique(failed_motion_ids).tolist():
            motion_mask = failed_motion_ids == motion_id
            if not torch.any(motion_mask):
                continue
            bin_count = int(self.motion_bin_counts[motion_id].item())
            sample_span = int(self.motion_sampling_max_starts[motion_id].item()) + 1
            current_bin_index = torch.clamp(
                (failed_time_steps[motion_mask] * bin_count) // max(sample_span, 1), 0, bin_count - 1
            )
            self._current_bin_failed[motion_id, :bin_count] = torch.bincount(current_bin_index, minlength=bin_count)

    def _smooth_sampling_probabilities(self, base_probabilities: torch.Tensor) -> torch.Tensor:
        sampling_probabilities = torch.nn.functional.pad(
            base_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        return sampling_probabilities / sampling_probabilities.sum()

    def _adaptive_sampling_legacy(self, env_ids: torch.Tensor) -> None:
        self._update_failure_statistics_legacy(env_ids)

        sampling_probabilities = self.bin_failed_count[0, : self.bin_count]
        sampling_probabilities = sampling_probabilities + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = self._smooth_sampling_probabilities(sampling_probabilities)

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][env_ids] = H_norm
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count

    def _adaptive_sampling_per_motion(self, env_ids: torch.Tensor) -> None:
        self._update_failure_statistics_per_motion(env_ids)
        env_motion_ids = self.motion_ids[env_ids]
        for motion_id in torch.unique(env_motion_ids).tolist():
            motion_env_ids = env_ids[env_motion_ids == motion_id]
            if motion_env_ids.numel() == 0:
                continue
            bin_count = int(self.motion_bin_counts[motion_id].item())
            sample_span = int(self.motion_sampling_max_starts[motion_id].item()) + 1

            sampling_probabilities = self.bin_failed_count[motion_id, :bin_count]
            sampling_probabilities = sampling_probabilities + self.cfg.adaptive_uniform_ratio / float(bin_count)
            sampling_probabilities = self._smooth_sampling_probabilities(sampling_probabilities)

            sampled_bins = torch.multinomial(sampling_probabilities, motion_env_ids.numel(), replacement=True)
            sampled_steps = torch.floor(
                (sampled_bins + torch.rand(motion_env_ids.numel(), device=self.device)) / bin_count * sample_span
            ).long()
            self.time_steps[motion_env_ids] = torch.clamp(sampled_steps, max=max(sample_span - 1, 0))

            H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
            H_norm = H / max(math.log(bin_count), 1.0)
            pmax, imax = sampling_probabilities.max(dim=0)
            self.metrics["sampling_entropy"][motion_env_ids] = H_norm
            self.metrics["sampling_top1_prob"][motion_env_ids] = pmax
            self.metrics["sampling_top1_bin"][motion_env_ids] = imax.float() / bin_count

    def _should_resample_motion_ids(self) -> bool:
        if self.num_motions == 1 or not self.cfg.lock_motion_per_episode:
            return False
        if self.motion_resample_scope == "episode_reset_only":
            return self._resample_reason == "episode_reset"
        return self._resample_reason != "unknown"

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.phase_sampling_strategy == "uniform":
            self._sample_uniform_time_steps(env_ids_tensor)
            return
        if self.phase_sampling_strategy == "adaptive_per_motion":
            self._adaptive_sampling_per_motion(env_ids_tensor)
            return
        if self.num_motions > 1:
            self._sample_uniform_time_steps(env_ids_tensor)
            return
        self._adaptive_sampling_legacy(env_ids_tensor)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        previous_motion_ids = self.motion_ids[env_ids].clone()
        previous_time_steps = self.time_steps[env_ids].clone()
        if self._should_resample_motion_ids():
            self.motion_ids[env_ids_tensor] = torch.multinomial(self.motion_prob, len(env_ids), replacement=True)
        self.motion_ended[env_ids_tensor] = False
        self._adaptive_sampling(env_ids_tensor)
        self._record_refpose_resample_events(env_ids_tensor, previous_motion_ids, previous_time_steps)
        self._refresh_motion_buffers(env_ids_tensor)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.motion_ended.zero_()
        self.time_steps += 1
        ended_env_ids = torch.where(self.time_steps >= self.motion_lengths[self.motion_ids])[0]
        if self.motion_end_behavior == "terminate_episode":
            if ended_env_ids.numel() > 0:
                self.motion_ended[ended_env_ids] = True
                self.time_steps[ended_env_ids] = torch.clamp(
                    self.motion_lengths[self.motion_ids[ended_env_ids]] - 1,
                    min=0,
                )
        else:
            self._resample_reason = "motion_rollover"
            self._resample_command(ended_env_ids)
            self._resample_reason = "unknown"
        self._refresh_motion_buffers()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _record_refpose_resample_events(
        self, env_ids: Sequence[int], previous_motion_ids: torch.Tensor, previous_time_steps: torch.Tensor
    ) -> None:
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        for local_idx, env_id_tensor in enumerate(env_ids_tensor):
            env_id = int(env_id_tensor.item())
            prev_motion_id = int(previous_motion_ids[local_idx].item())
            prev_time_step = int(previous_time_steps[local_idx].item())
            motion_id = int(self.motion_ids[env_id].item())
            time_step = int(self.time_steps[env_id].item())
            self._refpose_log_events.append(
                {
                    "env_id": env_id,
                    "reason": self._resample_reason,
                    "prev_motion_id": prev_motion_id,
                    "prev_time_step": prev_time_step,
                    "motion_id": motion_id,
                    "time_step": time_step,
                }
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | list[str] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    sampling_preset: str = "beyondmimic"
    motion_resample_scope: str | None = None
    phase_sampling_window: str | None = None
    phase_sampling_strategy: str | None = None
    motion_end_behavior: str | None = None
    motion_sampling: str = "uniform"
    motion_weights: list[float] | None = None
    lock_motion_per_episode: bool = True
    motion_storage_device: str | None = None

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
