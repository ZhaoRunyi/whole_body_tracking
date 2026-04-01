# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--render_refpose",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Render motion reference poses during training when not running headless.",
)
parser.add_argument(
    "--print_refpose",
    type=int,
    nargs="?",
    const=8,
    default=0,
    help="Print a snapshot of the current reference motion source for the first N environments. Disabled by default.",
)
parser.add_argument(
    "--registry_name",
    type=str,
    nargs="+",
    default=None,
    help="One or more wandb motion registries (space-separated).",
)
parser.add_argument(
    "--local_dir",
    type=str,
    default=None,
    help="Recursively read all *.npz motions from local directory.",
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
import torch
from datetime import datetime
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


REQUIRED_MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _configure_training_visualization(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, render_refpose: bool
) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None:
        motion_cfg.debug_vis = render_refpose and not bool(getattr(args_cli, "headless", False))

    contact_sensor_cfg = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact_sensor_cfg is not None:
        contact_sensor_cfg.debug_vis = False


def _maybe_print_refpose_snapshot(env: gym.Env, num_envs_to_print: int) -> None:
    if num_envs_to_print <= 0:
        return

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    max_envs = min(num_envs_to_print, motion_cmd.num_envs)
    if max_envs <= 0:
        return

    print(f"[INFO] Available reference motions ({motion_cmd.num_motions} total):")
    for motion_id, motion_source in enumerate(motion_cmd.motion_files):
        print(f"[INFO]   motion[{motion_id}] file={motion_source}")

    print(f"[INFO] Current reference motion snapshot for first {max_envs}/{motion_cmd.num_envs} envs:")
    for env_id in range(max_envs):
        motion_id = int(motion_cmd.motion_ids[env_id].item())
        time_step = int(motion_cmd.time_steps[env_id].item())
        motion_source = motion_cmd.motion_files[motion_id]
        print(f"[INFO]   env[{env_id}] -> motion[{motion_id}] step={time_step} file={motion_source}")

    if not motion_cmd.cfg.lock_motion_per_episode:
        print("[INFO] Reference motion snapshot is not sticky because lock_motion_per_episode=False.")


def _normalize_registry_names(registry_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in registry_names:
        name = name.strip()
        if not name:
            continue
        if ":" not in name:
            name += ":latest"
        out.append(name)
    if not out:
        raise ValueError("--registry_name is empty.")
    return out


def _iter_motion_npz_files(root: str) -> list[str]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(str(root_path))

    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".npz"):
                files.append(Path(dirpath) / filename)
    files.sort(key=lambda p: str(p))
    if not files:
        raise FileNotFoundError(f"No *.npz motion files found under: {root_path}")
    return [str(path) for path in files]


def _validate_motion_npz_file(path: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_MOTION_KEYS if k not in data]
    if missing:
        raise ValueError(f"Motion file {path} missing keys: {missing}")


def _download_motion_npz_list(registry_names: list[str]) -> list[str]:
    import wandb

    api = wandb.Api()
    out: list[str] = []
    for registry_name in registry_names:
        artifact = api.artifact(registry_name)
        motion_file = str(Path(artifact.download()) / "motion.npz")
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"motion.npz not found in artifact dir for {registry_name}")
        _validate_motion_npz_file(motion_file)
        out.append(motion_file)
    return out


def _resolve_motion_files() -> tuple[str | list[str], list[str]]:
    has_registry = bool(args_cli.registry_name)
    has_local_dir = bool(args_cli.local_dir)
    if has_registry == has_local_dir:
        raise ValueError("Provide exactly one of --registry_name or --local_dir.")

    if has_local_dir:
        motion_files = _iter_motion_npz_files(args_cli.local_dir)
        for motion_file in motion_files:
            _validate_motion_npz_file(motion_file)
        return (motion_files[0] if len(motion_files) == 1 else motion_files), []

    registry_names = _normalize_registry_names(args_cli.registry_name)
    motion_files = _download_motion_npz_list(registry_names)
    return (motion_files[0] if len(motion_files) == 1 else motion_files), registry_names


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # load one or more motion files from wandb registry or local recursive directory
    motion_file, registry_names = _resolve_motion_files()
    env_cfg.commands.motion.motion_file = motion_file
    _configure_training_visualization(env_cfg, args_cli.render_refpose)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _maybe_print_refpose_snapshot(env, args_cli.print_refpose)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_names
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
