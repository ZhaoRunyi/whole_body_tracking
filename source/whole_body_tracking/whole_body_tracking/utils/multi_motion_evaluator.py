from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

import isaaclab.utils.math as math_utils
import numpy as np
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
SUCCESS_REASON_KEYS = ("motion_end", "time_out")
PRIMARY_TERMINATION_REASON_ORDER = ("motion_end", "time_out", "anchor_pos", "anchor_ori", "ee_body_pos", "terminated")
DEFAULT_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


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


def _get_termination_param(base_env, term_name: str, param_name: str, default: Any) -> Any:
    cfg = getattr(base_env, "cfg", None)
    terminations_cfg = getattr(cfg, "terminations", None) if cfg is not None else None
    term_cfg = getattr(terminations_cfg, term_name, None) if terminations_cfg is not None else None
    params = getattr(term_cfg, "params", None)
    if isinstance(params, dict) and param_name in params:
        return params[param_name]
    return default


def _get_body_indexes(motion_command, body_names: list[str] | tuple[str, ...] | None) -> list[int]:
    if body_names is None:
        return list(range(len(motion_command.cfg.body_names)))
    selected = set(body_names)
    return [idx for idx, name in enumerate(motion_command.cfg.body_names) if name in selected]


def _tensor_to_float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]


def _build_joint_diagnostics(robot, motion_command, env_id: int) -> dict[str, Any]:
    ref_joint_pos = motion_command.joint_pos[env_id]
    actual_joint_pos = motion_command.robot_joint_pos[env_id]
    joint_diff = actual_joint_pos - ref_joint_pos
    abs_joint_diff = joint_diff.abs()

    joint_names = list(getattr(robot, "joint_names", []) or [])
    if len(joint_names) != int(ref_joint_pos.numel()):
        joint_names = [f"joint_{i}" for i in range(int(ref_joint_pos.numel()))]

    ranked = torch.argsort(abs_joint_diff, descending=True)
    top_joint_errors = []
    for joint_idx_tensor in ranked[: min(5, ranked.numel())]:
        joint_idx = int(joint_idx_tensor.item())
        top_joint_errors.append(
            {
                "joint_name": joint_names[joint_idx],
                "joint_index": joint_idx,
                "ref": float(ref_joint_pos[joint_idx].item()),
                "actual": float(actual_joint_pos[joint_idx].item()),
                "diff": float(joint_diff[joint_idx].item()),
                "abs_diff": float(abs_joint_diff[joint_idx].item()),
            }
        )

    return {
        "joint_names": joint_names,
        "ref_joint_pos": _tensor_to_float_list(ref_joint_pos),
        "actual_joint_pos": _tensor_to_float_list(actual_joint_pos),
        "joint_pos_diff": _tensor_to_float_list(joint_diff),
        "top_abs_joint_pos_errors": top_joint_errors,
    }


def _build_failure_details_for_env(base_env, motion_command, env_id: int, reason: str) -> dict[str, Any]:
    robot = base_env.scene["robot"]
    details: dict[str, Any] = {
        "reason": reason,
        "env_id": int(env_id),
        "motion_id": int(motion_command.motion_ids[env_id].item()),
        "time_step": int(motion_command.time_steps[env_id].item()),
        "joint_diagnostics": _build_joint_diagnostics(robot, motion_command, env_id),
    }

    if reason == "anchor_pos":
        threshold = float(_get_termination_param(base_env, "anchor_pos", "threshold", 0.25))
        ref_pos = motion_command.anchor_pos_w[env_id]
        actual_pos = motion_command.robot_anchor_pos_w[env_id]
        diff = actual_pos - ref_pos
        details["anchor_pos"] = {
            "threshold": threshold,
            "ref_pos_w": _tensor_to_float_list(ref_pos),
            "actual_pos_w": _tensor_to_float_list(actual_pos),
            "diff_xyz": _tensor_to_float_list(diff),
            "abs_diff_z": float(diff[-1].abs().item()),
            "diff_norm": float(torch.linalg.vector_norm(diff).item()),
        }
    elif reason == "anchor_ori":
        threshold = float(_get_termination_param(base_env, "anchor_ori", "threshold", 0.8))
        motion_projected_gravity_b = math_utils.quat_rotate_inverse(motion_command.anchor_quat_w, robot.data.GRAVITY_VEC_W)[env_id]
        robot_projected_gravity_b = math_utils.quat_rotate_inverse(motion_command.robot_anchor_quat_w, robot.data.GRAVITY_VEC_W)[env_id]
        details["anchor_ori"] = {
            "threshold": threshold,
            "ref_anchor_quat_w": _tensor_to_float_list(motion_command.anchor_quat_w[env_id]),
            "actual_anchor_quat_w": _tensor_to_float_list(motion_command.robot_anchor_quat_w[env_id]),
            "ref_projected_gravity_b": _tensor_to_float_list(motion_projected_gravity_b),
            "actual_projected_gravity_b": _tensor_to_float_list(robot_projected_gravity_b),
            "abs_diff_z": float((motion_projected_gravity_b[2] - robot_projected_gravity_b[2]).abs().item()),
            "quat_error_magnitude": float(math_utils.quat_error_magnitude(
                motion_command.anchor_quat_w[env_id : env_id + 1], motion_command.robot_anchor_quat_w[env_id : env_id + 1]
            )[0].item()),
        }
    elif reason == "ee_body_pos":
        threshold = float(_get_termination_param(base_env, "ee_body_pos", "threshold", 0.25))
        ee_body_names = _get_termination_param(base_env, "ee_body_pos", "body_names", DEFAULT_EE_BODY_NAMES)
        ee_body_indexes = _get_body_indexes(motion_command, ee_body_names)
        links = []
        violated_links = []
        worst_link = None
        worst_abs_diff_z = -1.0
        for body_index in ee_body_indexes:
            body_name = motion_command.cfg.body_names[body_index]
            ref_pos = motion_command.body_pos_relative_w[env_id, body_index]
            actual_pos = motion_command.robot_body_pos_w[env_id, body_index]
            diff = actual_pos - ref_pos
            abs_diff_z = float(diff[-1].abs().item())
            violated = abs_diff_z > threshold
            link_row = {
                "body_name": body_name,
                "body_index": int(body_index),
                "threshold": threshold,
                "ref_pos_w": _tensor_to_float_list(ref_pos),
                "actual_pos_w": _tensor_to_float_list(actual_pos),
                "diff_xyz": _tensor_to_float_list(diff),
                "abs_diff_z": abs_diff_z,
                "diff_norm": float(torch.linalg.vector_norm(diff).item()),
                "violated": violated,
            }
            links.append(link_row)
            if violated:
                violated_links.append(body_name)
            if abs_diff_z > worst_abs_diff_z:
                worst_abs_diff_z = abs_diff_z
                worst_link = link_row

        details["ee_body_pos"] = {
            "threshold": threshold,
            "body_names": list(ee_body_names),
            "violated_links": violated_links,
            "worst_link": worst_link,
            "links": links,
        }

    return details


def _compute_episode_reason_masks(base_env, motion_command, timeout_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    num_envs = int(motion_command.motion_ids.shape[0])
    device = motion_command.motion_ids.device

    motion_end_mask = motion_command.motion_ended.clone()

    anchor_pos_threshold = float(_get_termination_param(base_env, "anchor_pos", "threshold", 0.25))
    anchor_pos_mask = (
        torch.abs(motion_command.anchor_pos_w[:, -1] - motion_command.robot_anchor_pos_w[:, -1]) > anchor_pos_threshold
    )

    anchor_ori_threshold = float(_get_termination_param(base_env, "anchor_ori", "threshold", 0.8))
    robot = base_env.scene["robot"]
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(motion_command.anchor_quat_w, robot.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_rotate_inverse(
        motion_command.robot_anchor_quat_w, robot.data.GRAVITY_VEC_W
    )
    anchor_ori_mask = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > anchor_ori_threshold

    ee_body_threshold = float(_get_termination_param(base_env, "ee_body_pos", "threshold", 0.25))
    ee_body_names = _get_termination_param(base_env, "ee_body_pos", "body_names", DEFAULT_EE_BODY_NAMES)
    ee_body_indexes = _get_body_indexes(motion_command, ee_body_names)
    if ee_body_indexes:
        ee_body_error = torch.abs(
            motion_command.body_pos_relative_w[:, ee_body_indexes, -1] - motion_command.robot_body_pos_w[:, ee_body_indexes, -1]
        )
        ee_body_pos_mask = torch.any(ee_body_error > ee_body_threshold, dim=-1)
    else:
        ee_body_pos_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    return {
        "motion_end": motion_end_mask,
        "time_out": timeout_mask,
        "anchor_pos": anchor_pos_mask,
        "anchor_ori": anchor_ori_mask,
        "ee_body_pos": ee_body_pos_mask,
    }


def _select_reason_for_env(reason_masks: dict[str, torch.Tensor], env_id: int) -> str:
    reason = "terminated"
    for candidate in PRIMARY_TERMINATION_REASON_ORDER:
        if candidate == "terminated":
            continue
        candidate_mask = reason_masks.get(candidate)
        if candidate_mask is not None and bool(candidate_mask[env_id].item()):
            reason = candidate
            break
    return reason


def _classify_episode_outcomes(
    base_env, motion_command, timeout_mask: torch.Tensor, done_env_ids: torch.Tensor
) -> list[dict[str, Any]]:
    reason_masks = _compute_episode_reason_masks(base_env, motion_command, timeout_mask)
    outcomes: list[dict[str, Any]] = []
    for env_id_tensor in done_env_ids:
        env_id = int(env_id_tensor.item())
        reason = _select_reason_for_env(reason_masks, env_id)
        outcomes.append(
            {
                "success": reason in SUCCESS_REASON_KEYS,
                "termination_reason": reason,
            }
        )
    return outcomes


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


def _pin_motion_id(motion_command, motion_id: int) -> torch.Tensor:
    original_motion_prob = motion_command.motion_prob.clone()
    pinned_prob = torch.zeros_like(motion_command.motion_prob)
    pinned_prob[int(motion_id)] = 1.0
    motion_command.motion_prob = pinned_prob
    motion_command.motion_ids[:] = int(motion_id)
    return original_motion_prob


def _reset_env_if_possible(env) -> None:
    reset_fn = getattr(env, "reset", None)
    if reset_fn is None:
        return
    reset_output = reset_fn()
    if isinstance(reset_output, tuple):
        return


def _reset_env_ids_if_possible(env, base_env, env_ids: torch.Tensor) -> bool:
    if env_ids.numel() == 0:
        return True

    for reset_target in (base_env, getattr(env, "unwrapped", None)):
        if reset_target is None:
            continue
        reset_fn = getattr(reset_target, "reset", None)
        if reset_fn is None:
            continue
        try:
            reset_fn(env_ids=env_ids)
            return True
        except TypeError:
            continue

    if int(env_ids.numel()) == int(getattr(env, "num_envs", 0)):
        _reset_env_if_possible(env)
        return True
    return False


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
        self.success_count = [0 for _ in range(self.num_motions)]
        self.timeout_count = [0 for _ in range(self.num_motions)]
        self.episode_return_sum = [0.0 for _ in range(self.num_motions)]
        self.episode_length_sum = [0.0 for _ in range(self.num_motions)]
        self.metric_sum_by_name: dict[str, list[float]] = {}
        self.termination_reason_count_by_name: dict[str, list[int]] = {}

    def add_episode(
        self,
        motion_id: int,
        episode_return: float,
        episode_length: int,
        timed_out: bool,
        success: bool,
        termination_reason: str,
        mean_metrics: dict[str, float],
    ) -> int | None:
        if motion_id < 0 or motion_id >= self.num_motions:
            return None
        if self.episode_count[motion_id] >= self.target_episodes_per_motion:
            return None

        self.episode_count[motion_id] += 1
        if success:
            self.success_count[motion_id] += 1
        if timed_out:
            self.timeout_count[motion_id] += 1
        self.episode_return_sum[motion_id] += float(episode_return)
        self.episode_length_sum[motion_id] += float(episode_length)
        if termination_reason not in self.termination_reason_count_by_name:
            self.termination_reason_count_by_name[termination_reason] = [0 for _ in range(self.num_motions)]
        self.termination_reason_count_by_name[termination_reason][motion_id] += 1

        for metric_name, metric_value in mean_metrics.items():
            if metric_name not in self.metric_sum_by_name:
                self.metric_sum_by_name[metric_name] = [0.0 for _ in range(self.num_motions)]
            self.metric_sum_by_name[metric_name][motion_id] += float(metric_value)
        return self.episode_count[motion_id]

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
            success_count = self.success_count[motion_id]
            timeout_count = self.timeout_count[motion_id]
            early_termination_count = episode_count - success_count

            row: dict[str, Any] = {
                "motion_name": motion_name,
                "episodes": episode_count,
                "target_episodes": self.target_episodes_per_motion,
                "coverage": (episode_count / self.target_episodes_per_motion)
                if self.target_episodes_per_motion > 0
                else None,
                "successful_episodes": success_count,
                "failed_episodes": early_termination_count,
                "success_rate": (success_count / episode_count) if episode_count > 0 else None,
                "failure_rate": (early_termination_count / episode_count) if episode_count > 0 else None,
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
            for reason_name, reason_counts in sorted(self.termination_reason_count_by_name.items()):
                row[f"count_{reason_name}"] = reason_counts[motion_id]
                row[f"rate_{reason_name}"] = (reason_counts[motion_id] / episode_count) if episode_count > 0 else None
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
    pinned_motion_id: int | None = None,
    reset_env: bool = True,
    failure_hold_steps: int = 0,
    logical_timeout_steps: int | None = None,
    episode_start_callback=None,
    episode_callback=None,
    failure_frame_callback=None,
) -> dict[str, Any]:
    if target_episodes_per_motion <= 0:
        raise ValueError("target_episodes_per_motion must be > 0.")
    if max_steps is not None and max_steps <= 0:
        max_steps = None
    failure_hold_steps = max(int(failure_hold_steps), 0)
    if logical_timeout_steps is not None and logical_timeout_steps <= 0:
        logical_timeout_steps = None

    base_env = getattr(env, "unwrapped", env)
    motion_command = base_env.command_manager.get_term("motion")
    all_motion_files = list(motion_command.motion_files)
    if not all_motion_files:
        raise ValueError("No motion files are configured in env.commands.motion.motion_file.")
    all_motion_labels = make_motion_labels(all_motion_files)
    if pinned_motion_id is None:
        tracked_motion_ids = list(range(len(all_motion_files)))
    else:
        tracked_motion_ids = [int(pinned_motion_id)]
    tracked_motion_labels = [all_motion_labels[motion_id] for motion_id in tracked_motion_ids]
    tracked_motion_index = {motion_id: local_idx for local_idx, motion_id in enumerate(tracked_motion_ids)}

    expected_episode_length_steps = logical_timeout_steps or _resolve_expected_episode_length_steps(base_env)
    aggregator = _MotionEpisodeAggregator(
        motion_labels=tracked_motion_labels,
        target_episodes_per_motion=target_episodes_per_motion,
        expected_episode_length_steps=expected_episode_length_steps,
    )

    num_envs = int(env.num_envs)
    device = motion_command.motion_ids.device
    all_env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

    original_motion_prob = None
    if pinned_motion_id is not None:
        original_motion_prob = _pin_motion_id(motion_command, pinned_motion_id)

    if reset_env:
        _reset_env_if_possible(env)

    if force_full_motion_from_start:
        _force_motion_frame(base_env, motion_command, all_env_ids, time_step=0)

    obs, _ = env.get_observations()
    if episode_start_callback is not None:
        episode_start_callback()

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
    pending_failure = torch.zeros(num_envs, dtype=torch.bool, device=device)
    failure_reason_by_env = ["" for _ in range(num_envs)]
    failure_details_by_env: list[dict[str, Any] | None] = [None for _ in range(num_envs)]
    failure_frame_by_env: list[np.ndarray | None] = [None for _ in range(num_envs)]
    failure_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    failure_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    failure_metric_sums: dict[str, torch.Tensor] = {}
    previous_actions = None
    episode_rows: list[dict[str, Any]] = []

    total_steps = 0
    stop_reason = "simulation_stopped"

    while simulation_app.is_running():
        with torch.no_grad():
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
        env_done_mask = _to_bool_mask(dones, num_envs, device)
        env_timeout_mask = _extract_time_out_mask(info, num_envs, device)

        episode_return += reward_values
        episode_length += 1

        if logical_timeout_steps is None:
            logical_timeout_mask = env_timeout_mask
        else:
            logical_timeout_mask = episode_length >= int(logical_timeout_steps)
        timeout_mask = env_timeout_mask | logical_timeout_mask
        reason_masks = _compute_episode_reason_masks(base_env, motion_command, timeout_mask)

        failure_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for failure_reason in ("anchor_pos", "anchor_ori", "ee_body_pos"):
            failure_mask |= reason_masks.get(failure_reason, torch.zeros_like(failure_mask))
        new_failure_mask = failure_mask & (~pending_failure)
        new_failure_env_ids = new_failure_mask.nonzero(as_tuple=False).flatten()
        if new_failure_env_ids.numel() > 0:
            pending_failure[new_failure_env_ids] = True
            failure_length[new_failure_env_ids] = episode_length[new_failure_env_ids]
            failure_return[new_failure_env_ids] = episode_return[new_failure_env_ids]
            captured_failure_frame = None
            if failure_frame_callback is not None:
                try:
                    captured_failure_frame = failure_frame_callback()
                except Exception:
                    captured_failure_frame = None
            for metric_name, metric_accumulator in episode_metric_sums.items():
                if metric_name not in failure_metric_sums:
                    failure_metric_sums[metric_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
                failure_metric_sums[metric_name][new_failure_env_ids] = metric_accumulator[new_failure_env_ids]
            for env_id_tensor in new_failure_env_ids:
                env_id = int(env_id_tensor.item())
                failure_reason = _select_reason_for_env(reason_masks, env_id)
                failure_reason_by_env[env_id] = failure_reason
                failure_details_by_env[env_id] = _build_failure_details_for_env(base_env, motion_command, env_id, failure_reason)
                if captured_failure_frame is not None:
                    failure_frame_by_env[env_id] = np.asarray(captured_failure_frame).copy()

        held_failure_done_mask = pending_failure & ((episode_length - failure_length) >= failure_hold_steps)
        natural_success_mask = (reason_masks["motion_end"] | timeout_mask) & (~pending_failure)
        done_mask = env_done_mask | natural_success_mask | held_failure_done_mask
        done_env_ids = done_mask.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            done_motion_ids = current_motion_ids[done_env_ids].detach().cpu().tolist()
            done_timeouts = timeout_mask[done_env_ids].detach().cpu().tolist()

            done_returns: list[float] = []
            done_lengths: list[int] = []
            video_lengths: list[int] = []
            failure_detected_steps: list[int | None] = []
            done_outcomes: list[dict[str, Any]] = []
            metric_means_by_env: list[dict[str, float]] = []
            for local_idx, env_id_tensor in enumerate(done_env_ids):
                env_id = int(env_id_tensor.item())
                is_failed_episode = bool(pending_failure[env_id].item())
                if is_failed_episode:
                    episode_steps = max(int(failure_length[env_id].item()), 1)
                    done_returns.append(float(failure_return[env_id].item()))
                    done_lengths.append(episode_steps)
                    video_lengths.append(int(episode_length[env_id].item()))
                    failure_detected_steps.append(episode_steps)
                    done_outcomes.append(
                        {
                            "success": False,
                            "termination_reason": failure_reason_by_env[env_id] or "terminated",
                        }
                    )
                else:
                    episode_steps = max(int(episode_length[env_id].item()), 1)
                    done_returns.append(float(episode_return[env_id].item()))
                    done_lengths.append(episode_steps)
                    video_lengths.append(episode_steps)
                    failure_detected_steps.append(None)
                    reason = _select_reason_for_env(reason_masks, env_id)
                    done_outcomes.append(
                        {
                            "success": reason in SUCCESS_REASON_KEYS,
                            "termination_reason": reason,
                        }
                    )

                mean_metrics: dict[str, float] = {}
                metric_source = failure_metric_sums if is_failed_episode else episode_metric_sums
                for metric_name, metric_accumulator in metric_source.items():
                    mean_metrics[metric_name] = float(metric_accumulator[env_id].item() / episode_steps)
                metric_means_by_env.append(mean_metrics)

            for local_idx, motion_id in enumerate(done_motion_ids):
                if int(motion_id) not in tracked_motion_index:
                    continue
                env_id = int(done_env_ids[local_idx].item())
                local_motion_id = tracked_motion_index[int(motion_id)]
                outcome = done_outcomes[local_idx]
                episode_number = aggregator.add_episode(
                    motion_id=tracked_motion_index[int(motion_id)],
                    episode_return=float(done_returns[local_idx]),
                    episode_length=int(done_lengths[local_idx]),
                    timed_out=bool(done_timeouts[local_idx]),
                    success=bool(outcome["success"]),
                    termination_reason=str(outcome["termination_reason"]),
                    mean_metrics=metric_means_by_env[local_idx],
                )
                if episode_number is None:
                    continue
                episode_length_ratio = None
                if expected_episode_length_steps is not None and expected_episode_length_steps > 0:
                    episode_length_ratio = float(done_lengths[local_idx]) / float(expected_episode_length_steps)
                episode_row = {
                    "global_episode_index": len(episode_rows) + 1,
                    "motion_name": all_motion_labels[int(motion_id)],
                    "motion_file": os.path.abspath(all_motion_files[int(motion_id)]),
                    "motion_id": int(motion_id),
                    "episode_index_for_motion": int(episode_number),
                    "success": bool(outcome["success"]),
                    "termination_reason": str(outcome["termination_reason"]),
                    "episode_return": float(done_returns[local_idx]),
                    "episode_length_steps": int(done_lengths[local_idx]),
                    "episode_length_ratio": episode_length_ratio,
                    "video_length_steps": int(video_lengths[local_idx]),
                }
                if failure_detected_steps[local_idx] is not None:
                    episode_row["failure_detected_step"] = int(failure_detected_steps[local_idx])
                    episode_row["failure_hold_steps"] = int(video_lengths[local_idx] - done_lengths[local_idx])
                    if failure_details_by_env[env_id] is not None:
                        episode_row["failure_details"] = failure_details_by_env[env_id]
                    if failure_frame_by_env[env_id] is not None:
                        episode_row["_failure_frame_rgb"] = failure_frame_by_env[env_id]
                episode_rows.append(episode_row)
                status = "SUCCESS" if episode_row["success"] else "FAIL"
                print(
                    "[EVAL][EP] "
                    f"motion={episode_row['motion_name']} "
                    f"episode={episode_row['episode_index_for_motion']}/{target_episodes_per_motion} "
                    f"status={status} "
                    f"reason={episode_row['termination_reason']} "
                    f"length_steps={episode_row['episode_length_steps']} "
                    f"return={episode_row['episode_return']:.4f}"
                )
                if episode_callback is not None:
                    episode_callback(episode_row)

            episode_return[done_env_ids] = 0.0
            episode_length[done_env_ids] = 0
            pending_failure[done_env_ids] = False
            failure_length[done_env_ids] = 0
            failure_return[done_env_ids] = 0.0
            for env_id_tensor in done_env_ids:
                env_id = int(env_id_tensor.item())
                failure_reason_by_env[env_id] = ""
                failure_details_by_env[env_id] = None
                failure_frame_by_env[env_id] = None
            for metric_accumulator in episode_metric_sums.values():
                metric_accumulator[done_env_ids] = 0.0
            for metric_accumulator in failure_metric_sums.values():
                metric_accumulator[done_env_ids] = 0.0

            manual_done_env_ids = done_env_ids[(~env_done_mask[done_env_ids]).nonzero(as_tuple=False).flatten()]
            if manual_done_env_ids.numel() > 0:
                _reset_env_ids_if_possible(env, base_env, manual_done_env_ids)
            if force_full_motion_from_start:
                if pinned_motion_id is not None:
                    motion_command.motion_ids[done_env_ids] = int(pinned_motion_id)
                _force_motion_frame(base_env, motion_command, done_env_ids, time_step=0)
                obs, _ = env.get_observations()
            if episode_start_callback is not None and not aggregator.is_target_reached():
                episode_start_callback()

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
    result["episodes"] = episode_rows
    result["config"] = {
        "target_episodes_per_motion": int(target_episodes_per_motion),
        "max_steps": max_steps,
        "print_interval": int(print_interval),
        "force_full_motion_from_start": bool(force_full_motion_from_start),
        "pinned_motion_id": pinned_motion_id,
        "failure_hold_steps": int(failure_hold_steps),
        "logical_timeout_steps": logical_timeout_steps,
        "tracked_motion_ids": tracked_motion_ids,
        "motion_files": [os.path.abspath(all_motion_files[motion_id]) for motion_id in tracked_motion_ids],
    }
    if original_motion_prob is not None:
        motion_command.motion_prob = original_motion_prob
    return result


def save_multi_motion_summary(
    result: dict[str, Any], output_dir: str, prefix: str = "multi_motion_eval"
) -> tuple[str, str, str | None]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"{prefix}_{timestamp}.csv")
    episode_csv_path = None

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    rows = result.get("motions", [])
    ordered_keys = [
        "motion_name",
        "episodes",
        "target_episodes",
        "coverage",
        "successful_episodes",
        "failed_episodes",
        "success_rate",
        "failure_rate",
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

    episode_rows = result.get("episodes", [])
    if episode_rows:
        episode_csv_path = os.path.join(output_dir, f"{prefix}_episodes_{timestamp}.csv")
        episode_keys = [
            "global_episode_index",
            "motion_name",
            "motion_file",
            "motion_id",
            "episode_index_for_motion",
            "success",
            "termination_reason",
            "episode_return",
            "episode_length_steps",
            "episode_length_ratio",
        ]
        for row in episode_rows:
            for key in row:
                if key not in episode_keys:
                    episode_keys.append(key)
        with open(episode_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=episode_keys)
            writer.writeheader()
            for row in episode_rows:
                writer.writerow(row)

    return json_path, csv_path, episode_csv_path
