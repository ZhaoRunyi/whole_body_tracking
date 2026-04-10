"""Script to play or evaluate a checkpointed RSL-RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play or evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play or evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (aligned with train.py).")
parser.add_argument(
    "--render_refpose",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Render motion reference poses during play/eval when not running headless.",
)
parser.add_argument(
    "--sampling_strategy",
    type=str,
    default=None,
    help=(
        "Motion sampling preset or comma-separated overrides. Kept aligned with train.py. "
        "Ignored by default in full-motion evaluation."
    ),
)
parser.add_argument(
    "--registry_name",
    type=str,
    nargs="+",
    default=None,
    help="One or more wandb motion registries (space-separated), aligned with train.py.",
)
parser.add_argument(
    "--local_dir",
    type=str,
    default=None,
    help="Recursively read all *.npz motions from local directory, aligned with train.py.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Backward-compatible alias for a single motion path (or comma-separated list).",
)
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    default=None,
    help="Backward-compatible alias for space-separated and/or comma-separated motion file paths.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Backward-compatible alias for --local_dir.",
)
parser.add_argument(
    "--evaluate",
    action="store_true",
    default=False,
    help="Run quantitative multi-motion evaluation and exit.",
)
parser.add_argument(
    "--eval_episodes_per_motion",
    type=int,
    default=10,
    help="Target completed episodes per motion in evaluation mode.",
)
parser.add_argument(
    "--eval_max_steps",
    type=int,
    default=None,
    help="Hard cap for evaluation loop in environment steps. Omit to disable.",
)
parser.add_argument(
    "--eval_print_interval",
    type=int,
    default=200,
    help="Progress print interval (steps) in evaluation mode.",
)
parser.add_argument(
    "--eval_mode",
    type=str,
    choices=("grouped", "separate", "separate_reuse"),
    default="separate_reuse",
    help=(
        "`grouped` aggregates multi-motion evaluation in one env; "
        "`separate` evaluates each motion independently while reusing one env; "
        "`separate_reuse` is a backward-compatible alias of `separate`."
    ),
)
parser.add_argument(
    "--eval_full_motion",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When enabled, each evaluation episode starts from frame 0 of the motion and runs the whole motion.",
)
parser.add_argument(
    "--export_onnx",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export policy ONNX during play/eval. Disable if ONNX export fails in your environment.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp.commands import SAMPLING_PRESET_DEFAULTS
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.multi_motion_evaluator import evaluate_multi_motion_policy, save_multi_motion_summary


REQUIRED_MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

_EVAL_VIDEO_RENAMERS = {}
SHORT_EVAL_VIDEO_PAD_STEPS = 2
SHORT_EVAL_VIDEO_TAIL_SECONDS = 2.0

SAMPLING_STRATEGY_KEY_ALIASES = {
    "preset": "sampling_preset",
    "sampling_preset": "sampling_preset",
    "resample_scope": "motion_resample_scope",
    "motion_resample_scope": "motion_resample_scope",
    "window": "phase_sampling_window",
    "phase_sampling_window": "phase_sampling_window",
    "phase": "phase_sampling_strategy",
    "phase_sampling_strategy": "phase_sampling_strategy",
    "motion_end": "motion_end_behavior",
    "motion_end_behavior": "motion_end_behavior",
}


def _split_motion_tokens(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _iter_motion_npz_files(root: str) -> list[str]:
    root_path = pathlib.Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(str(root_path))
    files = sorted(str(path) for path in root_path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No *.npz motion files found under: {root_path}")
    return files


def _validate_motion_npz_file(path: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_MOTION_KEYS if key not in data]
    if missing:
        raise ValueError(f"Motion file {path} missing keys: {missing}")


def _normalize_registry_names(registry_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in registry_names:
        stripped = name.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            stripped += ":latest"
        out.append(stripped)
    if not out:
        raise ValueError("--registry_name is empty.")
    return out


def _download_motion_npz_list(registry_names: list[str]) -> list[str]:
    import wandb

    api = wandb.Api()
    out: list[str] = []
    for registry_name in registry_names:
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"motion.npz not found in artifact dir for {registry_name}")
        _validate_motion_npz_file(motion_file)
        out.append(str(pathlib.Path(motion_file).resolve()))
    return out


def _resolve_explicit_motion_selection() -> tuple[list[str], list[str]]:
    raw_values: list[str] = []
    if args_cli.motion_file:
        raw_values.append(args_cli.motion_file)
    if args_cli.motion_files:
        raw_values.extend(args_cli.motion_files)

    direct_motion_files = _split_motion_tokens(raw_values)
    local_dir = args_cli.local_dir if args_cli.local_dir is not None else args_cli.motion_dir
    has_direct = bool(direct_motion_files)
    has_local_dir = bool(local_dir)
    has_registry = bool(args_cli.registry_name)

    if sum((has_direct, has_local_dir, has_registry)) > 1:
        raise ValueError(
            "Provide only one motion source among --motion_file/--motion_files, --local_dir/--motion_dir, or --registry_name."
        )

    if has_local_dir:
        resolved_local = _iter_motion_npz_files(local_dir)
        for motion_file in resolved_local:
            _validate_motion_npz_file(motion_file)
        return resolved_local, []

    if has_registry:
        registry_names = _normalize_registry_names(args_cli.registry_name)
        return _download_motion_npz_list(registry_names), registry_names

    resolved: list[str] = []
    seen: set[str] = set()
    for motion_file in direct_motion_files:
        path = str(pathlib.Path(motion_file).expanduser().resolve())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Motion file does not exist: {path}")
        _validate_motion_npz_file(path)
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved, []


def _get_motion_file_list_from_cfg(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> list[str]:
    motion_files = env_cfg.commands.motion.motion_file
    if isinstance(motion_files, str):
        return [motion_files]
    return list(motion_files)


def _apply_motion_override(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, motion_files: list[str]
) -> None:
    if not motion_files:
        return
    env_cfg.commands.motion.motion_file = motion_files[0] if len(motion_files) == 1 else motion_files
    print(f"[INFO] Using {len(motion_files)} motion file(s) from CLI/artifact override.")
    for idx, motion_file in enumerate(motion_files):
        print(f"[INFO]   motion[{idx}]: {motion_file}")


def _parse_sampling_strategy_spec(spec: str | None) -> dict[str, str]:
    if spec is None:
        return {}

    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--sampling_strategy cannot be empty.")

    overrides: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            key = "sampling_preset"
            value = token
        else:
            raw_key, value = token.split("=", 1)
            key = SAMPLING_STRATEGY_KEY_ALIASES.get(raw_key.strip())
            value = value.strip()
            if key is None:
                valid_keys = ", ".join(sorted(SAMPLING_STRATEGY_KEY_ALIASES))
                raise ValueError(f"Unsupported --sampling_strategy key '{raw_key.strip()}'. Valid keys: {valid_keys}")
            if not value:
                raise ValueError(f"Missing value for --sampling_strategy key '{raw_key.strip()}'.")
        overrides[key] = value

    return overrides


def _apply_sampling_strategy(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, sampling_strategy_spec: str | None
) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None or sampling_strategy_spec is None:
        return

    overrides = _parse_sampling_strategy_spec(sampling_strategy_spec)
    for key, value in overrides.items():
        setattr(motion_cfg, key, value)

    preset_name = getattr(motion_cfg, "sampling_preset", "beyondmimic")
    preset_defaults = SAMPLING_PRESET_DEFAULTS[preset_name]
    resolved = {
        "sampling_preset": preset_name,
        "motion_resample_scope": getattr(motion_cfg, "motion_resample_scope", None)
        or preset_defaults["motion_resample_scope"],
        "phase_sampling_window": getattr(motion_cfg, "phase_sampling_window", None)
        or preset_defaults["phase_sampling_window"],
        "phase_sampling_strategy": getattr(motion_cfg, "phase_sampling_strategy", None)
        or preset_defaults["phase_sampling_strategy"],
        "motion_end_behavior": getattr(motion_cfg, "motion_end_behavior", None)
        or preset_defaults["motion_end_behavior"],
    }
    resolved_str = ", ".join(f"{key}={value}" for key, value in resolved.items())
    print(f"[INFO] Motion sampling strategy: {resolved_str}")


def _configure_motion_sampling_for_evaluation(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    force_full_motion: bool,
) -> None:
    motion_cfg = env_cfg.commands.motion
    motion_cfg.motion_sampling = "uniform"
    motion_cfg.lock_motion_per_episode = True
    motion_cfg.motion_resample_scope = "episode_reset_only"
    motion_cfg.phase_sampling_window = "full_motion" if force_full_motion else "truncate_to_episode"
    motion_cfg.phase_sampling_strategy = "uniform"
    motion_cfg.motion_end_behavior = "terminate_episode"


def _configure_play_visualization(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, render_refpose: bool
) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None:
        motion_cfg.debug_vis = render_refpose and not bool(getattr(args_cli, "headless", False))

    contact_sensor_cfg = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact_sensor_cfg is not None:
        contact_sensor_cfg.debug_vis = False


def _create_wrapped_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    log_dir: str,
    video_enabled: bool,
    video_mode: str = "play",
    eval_artifact_dir: str | None = None,
):
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if video_enabled else None)
    eval_video_renamer = None
    if video_enabled:
        if video_mode == "eval":
            if eval_artifact_dir is None:
                raise ValueError("eval_artifact_dir must be provided when recording evaluation videos.")
            eval_video_folder = os.path.join(eval_artifact_dir, "videos")
            video_kwargs = {
                "video_folder": eval_video_folder,
                "episode_trigger": lambda episode_id: True,
                "video_length": 0,
                "name_prefix": "eval",
                "disable_logger": True,
            }
            print("[INFO] Recording full-episode evaluation videos.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            eval_video_renamer = _EvalEpisodeVideoRenamer(env)
        else:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    wrapped_env = RslRlVecEnvWrapper(env)
    if eval_video_renamer is not None:
        _EVAL_VIDEO_RENAMERS[id(wrapped_env)] = eval_video_renamer
        try:
            wrapped_env._eval_episode_video_renamer = eval_video_renamer
        except AttributeError:
            pass
    return wrapped_env


def _sanitize_video_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "episode"


def _resolve_ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _pad_short_eval_video(video_path: pathlib.Path, tail_seconds: float = SHORT_EVAL_VIDEO_TAIL_SECONDS) -> bool:
    ffmpeg = _resolve_ffmpeg_executable()
    if ffmpeg is None:
        print(f"[WARN] Could not pad short eval video because ffmpeg was not found: {video_path}")
        return False

    tmp_path = video_path.with_name(f"{video_path.stem}.pad_tmp{video_path.suffix}")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={float(tail_seconds):.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        if tmp_path.exists():
            tmp_path.unlink()
        stderr_tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown ffmpeg error"
        print(f"[WARN] Could not pad short eval video {video_path}: {stderr_tail}")
        return False

    tmp_path.replace(video_path)
    return True


class _EvalEpisodeVideoRenamer:
    def __init__(self, record_video_wrapper):
        self.record_video_wrapper = record_video_wrapper
        self.video_folder = pathlib.Path(record_video_wrapper.video_folder)
        self.current_video_path: pathlib.Path | None = None
        self.current_metadata_path: pathlib.Path | None = None

    def start_episode(self) -> None:
        if not bool(args_cli.video):
            return

        # Keep RecordVideo's internal episode state aligned with the evaluator's episode boundary.
        self.record_video_wrapper.terminated = False
        self.record_video_wrapper.truncated = False
        if not self.record_video_wrapper.recording:
            self.record_video_wrapper.start_video_recorder()

        video_recorder = getattr(self.record_video_wrapper, "video_recorder", None)
        self.current_video_path = pathlib.Path(video_recorder.path) if video_recorder is not None else None
        metadata_path = getattr(video_recorder, "metadata_path", None) if video_recorder is not None else None
        self.current_metadata_path = pathlib.Path(metadata_path) if metadata_path is not None else None

    def finish_episode(self, episode_row: dict) -> None:
        motion_name = _sanitize_video_stem(str(episode_row.get("motion_name", "motion")))
        episode_idx = int(episode_row.get("episode_index_for_motion", 0))
        status = "success" if bool(episode_row.get("success", False)) else "fail"
        target_stem = f"{motion_name}_episode_{episode_idx:03d}_{status}"
        target_path = self.video_folder / f"{target_stem}.mp4"

        dedup_index = 1
        while target_path.exists():
            target_path = self.video_folder / f"{target_stem}_{dedup_index:02d}.mp4"
            dedup_index += 1

        source_path = self.current_video_path
        source_metadata_path = self.current_metadata_path
        if self.record_video_wrapper.recording:
            self.record_video_wrapper.close_video_recorder()

        if source_path is None or not source_path.exists():
            episode_row["video_file"] = None
            print(f"[WARN] Expected eval video was not written: {source_path}")
            self.current_video_path = None
            self.current_metadata_path = None
            return

        source_path.rename(target_path)
        if source_metadata_path is not None and source_metadata_path.exists():
            source_metadata_path.rename(target_path.with_suffix(".meta.json"))
        episode_row["video_file"] = str(target_path.resolve())

        episode_length_steps = int(episode_row.get("episode_length_steps", 0))
        if 0 < episode_length_steps <= SHORT_EVAL_VIDEO_PAD_STEPS:
            padded = _pad_short_eval_video(target_path)
            episode_row["video_padded_for_viewing"] = padded
            if padded:
                episode_row["video_padding_tail_seconds"] = SHORT_EVAL_VIDEO_TAIL_SECONDS

        self.current_video_path = None
        self.current_metadata_path = None


def _get_eval_episode_video_renamer(env):
    return getattr(env, "_eval_episode_video_renamer", None) or _EVAL_VIDEO_RENAMERS.get(id(env))


def _load_runner_and_policy(env, agent_cfg: RslRlOnPolicyRunnerCfg, resume_path: str):
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    return ppo_runner, policy


def _export_policy_artifacts(env, ppo_runner, resume_path: str) -> None:
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)


def _print_evaluation_summary(result: dict) -> None:
    summary = result.get("summary", {})
    print(
        "[INFO] Evaluation finished:"
        f" steps={summary.get('total_steps')},"
        f" stop_reason={summary.get('stop_reason')},"
        f" all_targets_reached={summary.get('all_targets_reached')}"
    )
    print("[INFO] Per-motion metrics:")
    for row in result.get("motions", []):
        motion_name = row.get("motion_name")
        episodes = row.get("episodes")
        target_episodes = row.get("target_episodes")
        success_rate = row.get("success_rate")
        episode_return = row.get("mean_episode_return")
        anchor_err = row.get("mean_error_anchor_pos")
        body_err = row.get("mean_error_body_pos")
        print(
            f"[INFO]   {motion_name}: episodes={episodes}/{target_episodes}, "
            f"success_rate={success_rate}, return={episode_return}, "
            f"anchor_pos_err={anchor_err}, body_pos_err={body_err}"
        )


def _make_eval_artifact_dir(log_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_artifact_dir = os.path.join(log_dir, "eval", timestamp)
    os.makedirs(eval_artifact_dir, exist_ok=True)
    return eval_artifact_dir


def _run_grouped_evaluation(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    resume_path: str,
    log_dir: str,
    eval_artifact_dir: str,
) -> dict:
    env = _create_wrapped_env(
        env_cfg,
        log_dir=log_dir,
        video_enabled=bool(args_cli.video),
        video_mode="eval",
        eval_artifact_dir=eval_artifact_dir,
    )
    try:
        ppo_runner, policy = _load_runner_and_policy(env, agent_cfg, resume_path)
        if args_cli.export_onnx:
            _export_policy_artifacts(env, ppo_runner, resume_path)
        video_renamer = _get_eval_episode_video_renamer(env)
        return evaluate_multi_motion_policy(
            env=env,
            policy=policy,
            simulation_app=simulation_app,
            target_episodes_per_motion=args_cli.eval_episodes_per_motion,
            max_steps=args_cli.eval_max_steps,
            print_interval=args_cli.eval_print_interval,
            force_full_motion_from_start=args_cli.eval_full_motion,
            episode_start_callback=video_renamer.start_episode if video_renamer is not None else None,
            episode_callback=video_renamer.finish_episode if video_renamer is not None else None,
        )
    finally:
        env.close()


def _run_separate_motion_evaluation(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    resume_path: str,
    log_dir: str,
    eval_artifact_dir: str,
) -> dict:
    motion_files = _get_motion_file_list_from_cfg(env_cfg)
    combined_rows: list[dict] = []
    total_steps = 0
    all_targets_reached = True
    exported = False

    for motion_file in motion_files:
        env_cfg.commands.motion.motion_file = motion_file
        env = _create_wrapped_env(
            env_cfg,
            log_dir=log_dir,
            video_enabled=bool(args_cli.video),
            video_mode="eval",
            eval_artifact_dir=eval_artifact_dir,
        )
        try:
            ppo_runner, policy = _load_runner_and_policy(env, agent_cfg, resume_path)
            if not exported and args_cli.export_onnx:
                _export_policy_artifacts(env, ppo_runner, resume_path)
                exported = True
            video_renamer = _get_eval_episode_video_renamer(env)
            motion_result = evaluate_multi_motion_policy(
                env=env,
                policy=policy,
                simulation_app=simulation_app,
                target_episodes_per_motion=args_cli.eval_episodes_per_motion,
                max_steps=args_cli.eval_max_steps,
                print_interval=args_cli.eval_print_interval,
                force_full_motion_from_start=args_cli.eval_full_motion,
                episode_start_callback=video_renamer.start_episode if video_renamer is not None else None,
                episode_callback=video_renamer.finish_episode if video_renamer is not None else None,
            )
        finally:
            env.close()

        total_steps += int(motion_result.get("summary", {}).get("total_steps", 0))
        all_targets_reached = all_targets_reached and bool(
            motion_result.get("summary", {}).get("all_targets_reached", False)
        )
        if motion_result.get("motions"):
            row = dict(motion_result["motions"][0])
            row["motion_file"] = motion_file
            combined_rows.append(row)

    return {
        "summary": {
            "total_steps": total_steps,
            "stop_reason": "targets_reached" if all_targets_reached else "partial_completion",
            "all_targets_reached": all_targets_reached,
            "expected_episode_length_steps": None,
            "eval_mode": "separate",
        },
        "motions": combined_rows,
    }


def _run_separate_motion_evaluation_reuse(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    resume_path: str,
    log_dir: str,
    eval_artifact_dir: str,
) -> dict:
    motion_files = _get_motion_file_list_from_cfg(env_cfg)
    combined_rows: list[dict] = []
    combined_episodes: list[dict] = []
    total_steps = 0
    all_targets_reached = True

    env = _create_wrapped_env(
        env_cfg,
        log_dir=log_dir,
        video_enabled=bool(args_cli.video),
        video_mode="eval",
        eval_artifact_dir=eval_artifact_dir,
    )
    try:
        ppo_runner, policy = _load_runner_and_policy(env, agent_cfg, resume_path)
        if args_cli.export_onnx:
            _export_policy_artifacts(env, ppo_runner, resume_path)
        video_renamer = _get_eval_episode_video_renamer(env)

        for motion_id, motion_file in enumerate(motion_files):
            motion_result = evaluate_multi_motion_policy(
                env=env,
                policy=policy,
                simulation_app=simulation_app,
                target_episodes_per_motion=args_cli.eval_episodes_per_motion,
                max_steps=args_cli.eval_max_steps,
                print_interval=args_cli.eval_print_interval,
                force_full_motion_from_start=args_cli.eval_full_motion,
                pinned_motion_id=motion_id,
                reset_env=True,
                episode_start_callback=video_renamer.start_episode if video_renamer is not None else None,
                episode_callback=video_renamer.finish_episode if video_renamer is not None else None,
            )

            total_steps += int(motion_result.get("summary", {}).get("total_steps", 0))
            all_targets_reached = all_targets_reached and bool(
                motion_result.get("summary", {}).get("all_targets_reached", False)
            )
            if motion_result.get("motions"):
                row = dict(motion_result["motions"][0])
                row["motion_file"] = motion_file
                combined_rows.append(row)
            for episode_row in motion_result.get("episodes", []):
                combined_episode_row = dict(episode_row)
                combined_episode_row["global_episode_index"] = len(combined_episodes) + 1
                combined_episodes.append(combined_episode_row)
    finally:
        env.close()

    result = {
        "summary": {
            "total_steps": total_steps,
            "stop_reason": "targets_reached" if all_targets_reached else "partial_completion",
            "all_targets_reached": all_targets_reached,
            "expected_episode_length_steps": None,
            "eval_mode": "separate",
        },
        "motions": combined_rows,
    }
    if combined_episodes:
        result["episodes"] = combined_episodes
    return result


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play or evaluate with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    explicit_motion_files, registry_names = _resolve_explicit_motion_selection()
    artifact_motion_files: list[str] = []

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((artifact for artifact in wandb_run.used_artifacts() if artifact.type == "motions"), None)
        if art is None:
            print("[WARN] No motion artifact found in the run.")
        else:
            artifact_motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            if os.path.isfile(artifact_motion_file):
                artifact_motion_files = [str(pathlib.Path(artifact_motion_file).resolve())]
            else:
                print(f"[WARN] motion.npz not found in artifact directory: {artifact_motion_file}")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    selected_motion_files = explicit_motion_files if explicit_motion_files else artifact_motion_files
    _apply_motion_override(env_cfg, selected_motion_files)
    configured_motion_files = _get_motion_file_list_from_cfg(env_cfg)

    if args_cli.evaluate:
        _configure_motion_sampling_for_evaluation(env_cfg, force_full_motion=args_cli.eval_full_motion)
        if args_cli.sampling_strategy and args_cli.eval_full_motion:
            print("[WARN] --sampling_strategy is ignored when --eval_full_motion is enabled.")
        elif args_cli.sampling_strategy:
            _apply_sampling_strategy(env_cfg, args_cli.sampling_strategy)
    elif args_cli.sampling_strategy:
        _apply_sampling_strategy(env_cfg, args_cli.sampling_strategy)
    _configure_play_visualization(env_cfg, args_cli.render_refpose)

    video_enabled = bool(args_cli.video)
    if args_cli.evaluate and video_enabled and env_cfg.scene.num_envs != 1:
        print(
            "[WARN] Evaluation video captures the shared scene across environments. "
            "Use --num_envs=1 if you want one clean per-episode video stream."
        )

    log_dir = os.path.dirname(resume_path)

    if args_cli.evaluate:
        resolved_eval_mode = "separate" if args_cli.eval_mode in ("separate", "separate_reuse") else "grouped"
        eval_artifact_dir = _make_eval_artifact_dir(log_dir)
        if resolved_eval_mode == "separate":
            result = _run_separate_motion_evaluation_reuse(
                env_cfg, agent_cfg, resume_path, log_dir, eval_artifact_dir
            )
        else:
            result = _run_grouped_evaluation(env_cfg, agent_cfg, resume_path, log_dir, eval_artifact_dir)
        result.setdefault("config", {})
        result["config"].update(
            {
                "task": args_cli.task,
                "checkpoint_path": str(pathlib.Path(resume_path).resolve()),
                "wandb_path": args_cli.wandb_path,
                "registry_name": registry_names,
                "motion_files": configured_motion_files,
                "eval_mode": resolved_eval_mode,
                "eval_full_motion": args_cli.eval_full_motion,
                "eval_artifact_dir": str(pathlib.Path(eval_artifact_dir).resolve()),
            }
        )

        json_path, csv_path, episode_csv_path = save_multi_motion_summary(result=result, output_dir=eval_artifact_dir)
        _print_evaluation_summary(result)
        print(f"[INFO] Evaluation artifacts directory: {eval_artifact_dir}")
        print(f"[INFO] Saved evaluation JSON: {json_path}")
        print(f"[INFO] Saved evaluation CSV: {csv_path}")
        if episode_csv_path is not None:
            print(f"[INFO] Saved evaluation episode CSV: {episode_csv_path}")
    else:
        env = _create_wrapped_env(env_cfg, log_dir=log_dir, video_enabled=video_enabled)
        try:
            ppo_runner, policy = _load_runner_and_policy(env, agent_cfg, resume_path)
            if args_cli.export_onnx:
                _export_policy_artifacts(env, ppo_runner, resume_path)

            obs, _ = env.get_observations()
            timestep = 0
            while simulation_app.is_running():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                if video_enabled:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
        finally:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
