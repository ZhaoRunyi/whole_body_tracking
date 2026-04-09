from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

import torch


TRACKING_METRIC_KEYS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)

TIMEOUT_INFO_KEYS = ("time_outs", "time_out", "timeouts")
JOINT_EFFORT_ATTR_KEYS = ("applied_torque", "computed_torque", "joint_torque", "joint_torques", "applied_joint_efforts")
CONTACT_FORCE_ATTR_KEYS = ("net_forces_w", "net_forces_world", "net_forces_w_history")


def make_motion_labels(motion_files: list[str]) -> list[str]:
    stems = [os.path.splitext(os.path.basename(path))[0] for path in motion_files]
    if len(set(stems)) == len(stems):
        return stems

    basenames = [os.path.basename(path) for path in motion_files]
    if len(set(basenames)) == len(basenames):
        return basenames

    try:
        common_root = os.path.commonpath(motion_files)
    except ValueError:
        common_root = ""
    if common_root:
        relpaths = [os.path.relpath(path, common_root) for path in motion_files]
        if len(set(relpaths)) == len(relpaths):
            return relpaths

    return [os.path.abspath(path) for path in motion_files]


def _flatten_metric(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    values_flat = values.reshape(values.shape[0], -1)
    if values_flat.dtype == torch.bool:
        return values_flat.any(dim=-1)
    return values_flat.mean(dim=-1)


def _to_float_tensor(values: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device)
    else:
        tensor = torch.as_tensor(values, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    elif tensor.shape[0] != num_envs:
        tensor = tensor.reshape(num_envs, -1)
    tensor = _flatten_metric(tensor)
    return tensor.float()


def _to_bool_mask(values: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device)
    else:
        tensor = torch.as_tensor(values, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    elif tensor.shape[0] != num_envs:
        tensor = tensor.reshape(num_envs, -1)
    tensor = _flatten_metric(tensor)
    return tensor.bool()


def _extract_time_out_mask(info: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    if not isinstance(info, dict):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    for key in TIMEOUT_INFO_KEYS:
        if key in info:
            return _to_bool_mask(info[key], num_envs, device)
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _per_env_l2_norm(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.abs()
    return torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=-1)


def _extract_joint_effort_metric(robot) -> torch.Tensor | None:
    robot_data = getattr(robot, "data", None)
    if robot_data is None:
        return None
    for key in JOINT_EFFORT_ATTR_KEYS:
        values = getattr(robot_data, key, None)
        if isinstance(values, torch.Tensor):
            return _per_env_l2_norm(values)
    return None


def _extract_contact_force_metric(contact_sensor) -> torch.Tensor | None:
    sensor_data = getattr(contact_sensor, "data", None)
    if sensor_data is None:
        return None

    forces = None
    for key in CONTACT_FORCE_ATTR_KEYS:
        values = getattr(sensor_data, key, None)
        if isinstance(values, torch.Tensor):
            forces = values
            break
    if forces is None:
        return None
    if forces.ndim == 4:
        # [num_envs, history, bodies, 3] -> keep latest history slot
        forces = forces[:, -1]
    if forces.ndim < 3:
        return None
    force_magnitude = torch.linalg.vector_norm(forces, dim=-1)
    return force_magnitude.mean(dim=-1)


def _resolve_expected_episode_length_steps(base_env) -> int | None:
    for attr_name in ("max_episode_length", "max_episode_length_steps"):
        value = getattr(base_env, attr_name, None)
        if value is None:
            continue
        value_int = int(value)
        if value_int > 0:
            return value_int

    cfg = getattr(base_env, "cfg", None)
    step_dt = getattr(base_env, "step_dt", None)
    episode_length_s = getattr(cfg, "episode_length_s", None) if cfg is not None else None
    if step_dt is None or episode_length_s is None:
        return None

    step_dt_value = float(step_dt)
    if step_dt_value <= 0.0:
        return None
    return max(int(round(float(episode_length_s) / step_dt_value)), 1)


def _force_motion_frame(base_env, motion_command, env_ids, time_step: int = 0) -> None:
    env_ids_tensor = torch.as_tensor(env_ids, device=motion_command.device, dtype=torch.long)
    if env_ids_tensor.numel() == 0:
        return

    clamped_time_step = max(int(time_step), 0)
    max_time_steps = torch.clamp(motion_command.motion_lengths[motion_command.motion_ids[env_ids_tensor]] - 1, min=0)
    motion_command.motion_ended[env_ids_tensor] = False
    target_steps = torch.full_like(env_ids_tensor, clamped_time_step)
    target_steps = torch.minimum(target_steps, max_time_steps)
    motion_command.time_steps[env_ids_tensor] = target_steps
    motion_command._refresh_motion_buffers(env_ids_tensor)

    root_pos = motion_command.body_pos_w[:, 0].clone()
    root_ori = motion_command.body_quat_w[:, 0].clone()
    root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()
    joint_pos = motion_command.joint_pos.clone()
    joint_vel = motion_command.joint_vel.clone()

    motion_command.robot.write_joint_state_to_sim(joint_pos[env_ids_tensor], joint_vel[env_ids_tensor], env_ids=env_ids_tensor)
    motion_command.robot.write_root_state_to_sim(
        torch.cat(
            [
                root_pos[env_ids_tensor],
                root_ori[env_ids_tensor],
                root_lin_vel[env_ids_tensor],
                root_ang_vel[env_ids_tensor],
            ],
            dim=-1,
        ),
        env_ids=env_ids_tensor,
    )


def _collect_step_metrics(
    motion_command,
    actions: torch.Tensor,
    previous_actions: torch.Tensor | None,
    robot,
    contact_sensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}

    for key in TRACKING_METRIC_KEYS:
        value = motion_command.metrics.get(key)
        if isinstance(value, torch.Tensor):
            metrics[key] = value

    metrics["action_l2"] = _per_env_l2_norm(actions)
    if previous_actions is None:
        metrics["action_rate_l2"] = torch.zeros(actions.shape[0], device=actions.device)
    else:
        metrics["action_rate_l2"] = _per_env_l2_norm(actions - previous_actions)

    joint_effort_metric = _extract_joint_effort_metric(robot)
    if joint_effort_metric is not None:
        metrics["joint_effort_l2"] = joint_effort_metric

    contact_force_metric = _extract_contact_force_metric(contact_sensor)
    if contact_force_metric is not None:
        metrics["mean_contact_force"] = contact_force_metric

    return metrics


class _MotionEpisodeAggregator:
    def __init__(
        self,
        motion_labels: list[str],
        target_episodes_per_motion: int,
        expected_episode_length_steps: int | None,
    ):
        self.motion_labels = list(motion_labels)
        self.num_motions = len(self.motion_labels)
        self.target_episodes_per_motion = int(target_episodes_per_motion)
        self.expected_episode_length_steps = expected_episode_length_steps

        self.episode_count = [0 for _ in range(self.num_motions)]
        self.timeout_count = [0 for _ in range(self.num_motions)]
        self.episode_return_sum = [0.0 for _ in range(self.num_motions)]
        self.episode_length_sum = [0.0 for _ in range(self.num_motions)]
        self.metric_sum_by_name: dict[str, list[float]] = {}

    def add_episode(
        self,
        motion_id: int,
        episode_return: float,
        episode_length: int,
        timed_out: bool,
        mean_metrics: dict[str, float],
    ) -> None:
        if motion_id < 0 or motion_id >= self.num_motions:
            return

        self.episode_count[motion_id] += 1
        if timed_out:
            self.timeout_count[motion_id] += 1
        self.episode_return_sum[motion_id] += float(episode_return)
        self.episode_length_sum[motion_id] += float(episode_length)

        for metric_name, metric_value in mean_metrics.items():
            if metric_name not in self.metric_sum_by_name:
                self.metric_sum_by_name[metric_name] = [0.0 for _ in range(self.num_motions)]
            self.metric_sum_by_name[metric_name][motion_id] += float(metric_value)

    def is_target_reached(self) -> bool:
        return all(count >= self.target_episodes_per_motion for count in self.episode_count)

    def progress_line(self) -> str:
        return ", ".join(
            f"{self.motion_labels[idx]}: {self.episode_count[idx]}/{self.target_episodes_per_motion}"
            for idx in range(self.num_motions)
        )

    def build_summary(self, total_steps: int, stop_reason: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for motion_id, motion_name in enumerate(self.motion_labels):
            episode_count = self.episode_count[motion_id]
            timeout_count = self.timeout_count[motion_id]
            early_termination_count = episode_count - timeout_count

            row: dict[str, Any] = {
                "motion_name": motion_name,
                "episodes": episode_count,
                "target_episodes": self.target_episodes_per_motion,
                "coverage": (episode_count / self.target_episodes_per_motion)
                if self.target_episodes_per_motion > 0
                else None,
                "timeout_rate": (timeout_count / episode_count) if episode_count > 0 else None,
                "early_termination_rate": (early_termination_count / episode_count) if episode_count > 0 else None,
                "mean_episode_return": (self.episode_return_sum[motion_id] / episode_count) if episode_count > 0 else None,
                "mean_episode_length_steps": (self.episode_length_sum[motion_id] / episode_count)
                if episode_count > 0
                else None,
            }
            if self.expected_episode_length_steps is not None and episode_count > 0:
                row["mean_episode_length_ratio"] = row["mean_episode_length_steps"] / float(
                    self.expected_episode_length_steps
                )

            for metric_name, metric_sums in sorted(self.metric_sum_by_name.items()):
                row[f"mean_{metric_name}"] = (metric_sums[motion_id] / episode_count) if episode_count > 0 else None
            rows.append(row)

        return {
            "summary": {
                "total_steps": int(total_steps),
                "stop_reason": stop_reason,
                "all_targets_reached": self.is_target_reached(),
                "expected_episode_length_steps": self.expected_episode_length_steps,
            },
            "motions": rows,
        }


def evaluate_multi_motion_policy(
    env,
    policy,
    simulation_app,
    target_episodes_per_motion: int,
    max_steps: int | None = None,
    print_interval: int = 200,
    force_full_motion_from_start: bool = False,
) -> dict[str, Any]:
    if target_episodes_per_motion <= 0:
        raise ValueError("target_episodes_per_motion must be > 0.")
    if max_steps is not None and max_steps <= 0:
        max_steps = None

    base_env = getattr(env, "unwrapped", env)
    motion_command = base_env.command_manager.get_term("motion")
    motion_files = list(motion_command.motion_files)
    if not motion_files:
        raise ValueError("No motion files are configured in env.commands.motion.motion_file.")
    motion_labels = make_motion_labels(motion_files)

    expected_episode_length_steps = _resolve_expected_episode_length_steps(base_env)
    aggregator = _MotionEpisodeAggregator(
        motion_labels=motion_labels,
        target_episodes_per_motion=target_episodes_per_motion,
        expected_episode_length_steps=expected_episode_length_steps,
    )

    num_envs = int(env.num_envs)
    device = motion_command.motion_ids.device
    all_env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

    if force_full_motion_from_start:
        _force_motion_frame(base_env, motion_command, all_env_ids, time_step=0)

    obs, _ = env.get_observations()

    robot = base_env.scene["robot"]
    sensors = getattr(base_env.scene, "sensors", None)
    contact_sensor = None
    if isinstance(sensors, dict):
        contact_sensor = sensors.get("contact_forces")
    elif sensors is not None and hasattr(sensors, "get"):
        contact_sensor = sensors.get("contact_forces")
    elif sensors is not None and hasattr(sensors, "__contains__") and "contact_forces" in sensors:
        contact_sensor = sensors["contact_forces"]

    episode_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_metric_sums: dict[str, torch.Tensor] = {}
    previous_actions = None

    total_steps = 0
    stop_reason = "simulation_stopped"

    while simulation_app.is_running():
        with torch.inference_mode():
            current_motion_ids = motion_command.motion_ids.clone()
            actions = policy(obs)
            step_metrics = _collect_step_metrics(motion_command, actions, previous_actions, robot, contact_sensor)
            if not episode_metric_sums:
                episode_metric_sums = {
                    metric_name: torch.zeros(num_envs, dtype=torch.float32, device=device)
                    for metric_name in step_metrics
                }
            for metric_name, metric_values in step_metrics.items():
                if metric_name not in episode_metric_sums:
                    episode_metric_sums[metric_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
                episode_metric_sums[metric_name] += metric_values

            obs, rewards, dones, info = env.step(actions)

        reward_values = _to_float_tensor(rewards, num_envs, device)
        done_mask = _to_bool_mask(dones, num_envs, device)
        timeout_mask = _extract_time_out_mask(info, num_envs, device)

        episode_return += reward_values
        episode_length += 1

        done_env_ids = done_mask.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            done_motion_ids = current_motion_ids[done_env_ids].detach().cpu().tolist()
            done_returns = episode_return[done_env_ids].detach().cpu().tolist()
            done_lengths = episode_length[done_env_ids].detach().cpu().tolist()
            done_timeouts = timeout_mask[done_env_ids].detach().cpu().tolist()

            metric_means_by_env: list[dict[str, float]] = []
            for local_idx, env_id_tensor in enumerate(done_env_ids):
                env_id = int(env_id_tensor.item())
                episode_steps = max(int(done_lengths[local_idx]), 1)
                mean_metrics: dict[str, float] = {}
                for metric_name, metric_accumulator in episode_metric_sums.items():
                    mean_metrics[metric_name] = float(metric_accumulator[env_id].item() / episode_steps)
                metric_means_by_env.append(mean_metrics)

            for local_idx, motion_id in enumerate(done_motion_ids):
                aggregator.add_episode(
                    motion_id=int(motion_id),
                    episode_return=float(done_returns[local_idx]),
                    episode_length=int(done_lengths[local_idx]),
                    timed_out=bool(done_timeouts[local_idx]),
                    mean_metrics=metric_means_by_env[local_idx],
                )

            episode_return[done_env_ids] = 0.0
            episode_length[done_env_ids] = 0
            for metric_accumulator in episode_metric_sums.values():
                metric_accumulator[done_env_ids] = 0.0

            if force_full_motion_from_start:
                _force_motion_frame(base_env, motion_command, done_env_ids, time_step=0)
                obs, _ = env.get_observations()

        previous_actions = actions.detach()
        total_steps += 1

        if print_interval > 0 and total_steps % print_interval == 0:
            print(f"[EVAL] step={total_steps} | {aggregator.progress_line()}")

        if aggregator.is_target_reached():
            stop_reason = "targets_reached"
            break
        if max_steps is not None and total_steps >= max_steps:
            stop_reason = "max_steps_reached"
            break

    result = aggregator.build_summary(total_steps=total_steps, stop_reason=stop_reason)
    result["config"] = {
        "target_episodes_per_motion": int(target_episodes_per_motion),
        "max_steps": max_steps,
        "print_interval": int(print_interval),
        "force_full_motion_from_start": bool(force_full_motion_from_start),
        "motion_files": [os.path.abspath(path) for path in motion_files],
    }
    return result


def save_multi_motion_summary(
    result: dict[str, Any], output_dir: str, prefix: str = "multi_motion_eval"
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"{prefix}_{timestamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    rows = result.get("motions", [])
    ordered_keys = [
        "motion_name",
        "episodes",
        "target_episodes",
        "coverage",
        "timeout_rate",
        "early_termination_rate",
        "mean_episode_return",
        "mean_episode_length_steps",
        "mean_episode_length_ratio",
    ]
    all_keys = list(ordered_keys)
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path
