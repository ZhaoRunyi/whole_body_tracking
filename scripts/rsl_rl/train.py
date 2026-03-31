# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path
import subprocess
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
import torch.nn.functional as F
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


def _parse_cuda_version(version_str: str | None) -> tuple[int, int] | None:
    if not version_str:
        return None
    parts = version_str.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def _log_cuda_memory(stage: str, device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    max_allocated = torch.cuda.max_memory_allocated(device_index)
    max_reserved = torch.cuda.max_memory_reserved(device_index)
    print(
        f"[INFO] CUDA memory @ {stage}: "
        f"allocated={_format_gib(allocated)}, reserved={_format_gib(reserved)}, "
        f"max_allocated={_format_gib(max_allocated)}, max_reserved={_format_gib(max_reserved)}, "
        f"total={_format_gib(props.total_memory)} on {props.name}"
    )


def _log_system_gpu_memory(stage: str, device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type != "cuda":
        return

    device_index = device.index if device.index is not None else 0

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(
            f"[INFO] System GPU memory @ {stage}: "
            f"used={_format_gib(mem.used)}, free={_format_gib(mem.free)}, total={_format_gib(mem.total)}"
        )
        return
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used_mib, total_mib = [part.strip() for part in result.stdout.strip().split(",", maxsplit=1)]
        used_bytes = int(used_mib) * 1024 * 1024
        total_bytes = int(total_mib) * 1024 * 1024
        print(
            f"[INFO] System GPU memory @ {stage}: "
            f"used={_format_gib(used_bytes)}, total={_format_gib(total_bytes)}"
        )
    except Exception:
        pass


def _maybe_sync_cuda(stage: str, device: str | torch.device) -> None:
    if os.getenv("WBT_SYNC_CUDA_DEBUG") != "1":
        return

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(device_index)
    print(f"[INFO] CUDA synchronize passed @ {stage}")


def _log_torch_cuda_stack(device: str | torch.device) -> None:
    device = torch.device(device)
    print(f"[INFO] Torch stack: torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    arch_list = []
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        pass
    print(
        f"[INFO] CUDA device capability: sm_{capability[0]}{capability[1]} "
        f"on {torch.cuda.get_device_name(device_index)}; "
        f"torch arch list={arch_list}"
    )


def _log_python_env_stack() -> None:
    conda_env = os.getenv("CONDA_DEFAULT_ENV")
    conda_prefix = os.getenv("CONDA_PREFIX")
    conda_shlvl = os.getenv("CONDA_SHLVL")
    print(
        "[INFO] Python env stack: "
        f"sys.executable={sys.executable}, "
        f"CONDA_DEFAULT_ENV={conda_env!r}, CONDA_PREFIX={conda_prefix!r}, CONDA_SHLVL={conda_shlvl!r}"
    )
    try:
        shlvl = int(conda_shlvl) if conda_shlvl is not None else 0
    except ValueError:
        shlvl = 0
    if shlvl > 1:
        print(
            "[WARN] Detected stacked conda environments (CONDA_SHLVL > 1). "
            "This can mix shared libraries across envs even when `which python` looks correct."
        )


def _check_blackwell_torch_compatibility(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    torch_cuda_version = _parse_cuda_version(torch.version.cuda)

    # NVIDIA documents Blackwell support starting with CUDA 12.8.
    # Isaac Lab 2.1.0 also recommends cu128 nightly for 50-series GPUs
    # instead of the default torch 2.5.1/cu121 stack bundled with Isaac Sim.
    is_probably_blackwell = capability[0] >= 10 or "blackwell" in device_name.lower()
    if is_probably_blackwell and (torch_cuda_version is None or torch_cuda_version < (12, 8)):
        raise RuntimeError(
            "Detected a likely Blackwell GPU "
            f"({device_name}, sm_{capability[0]}{capability[1]}) with torch CUDA "
            f"{torch.version.cuda!r}. This stack is likely incompatible. "
            "Official NVIDIA CUDA docs add Blackwell support in CUDA 12.8, and Isaac Lab 2.1.0 "
            "explicitly recommends a cu128/nightly PyTorch build for 50-series GPUs instead of "
            "the default torch 2.5.1/cu121 install. Please upgrade the remote environment to a "
            "PyTorch build with CUDA 12.8+ (or newer) before debugging this repo further."
        )


def _run_cublas_smoke_test(stage: str, device: str | torch.device, batch_size: int = 4) -> None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    x = torch.randn(batch_size, 160, device=device, dtype=torch.float32)
    weight = torch.randn(512, 160, device=device, dtype=torch.float32)
    bias = torch.randn(512, device=device, dtype=torch.float32)
    y = F.linear(x, weight, bias)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(device_index)
    print(
        f"[INFO] cuBLAS smoke test passed @ {stage}: "
        f"input_shape={tuple(x.shape)}, output_shape={tuple(y.shape)}"
    )


def _extract_privileged_obs(extras: object) -> torch.Tensor | None:
    if not isinstance(extras, dict):
        return None
    observations = extras.get("observations")
    if not isinstance(observations, dict):
        return None
    critic_obs = observations.get("critic")
    return critic_obs if isinstance(critic_obs, torch.Tensor) else None


def _log_tensor_summary(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        print(f"[INFO] Tensor summary: {name}=None")
        return
    finite = bool(torch.isfinite(tensor).all().item())
    print(
        f"[INFO] Tensor summary: {name}.shape={tuple(tensor.shape)}, "
        f"{name}.device={tensor.device}, {name}.dtype={tensor.dtype}, finite={finite}"
    )


def _preflight_env_and_policy(env, runner, device: str | torch.device) -> None:
    print("[INFO] Running env/policy preflight before learn...")
    obs, extras = env.get_observations()
    _maybe_sync_cuda("after get_observations preflight", device)
    privileged_obs = _extract_privileged_obs(extras)
    _log_tensor_summary("policy_obs", obs)
    _log_tensor_summary("critic_obs", privileged_obs)

    with torch.no_grad():
        _ = runner.alg.act(obs, privileged_obs)
    _maybe_sync_cuda("after policy act preflight", device)
    print("[INFO] Env/policy preflight passed before learn.")


def _maybe_run_single_step_preflight(env, runner, device: str | torch.device) -> None:
    if os.getenv("WBT_STEP_PREFLIGHT") != "1":
        return

    print("[INFO] Running single-step rollout preflight before learn...")
    obs, extras = env.get_observations()
    privileged_obs = _extract_privileged_obs(extras)
    with torch.no_grad():
        actions = runner.alg.act(obs, privileged_obs)
    _maybe_sync_cuda("after rollout preflight act", device)
    _log_tensor_summary("preflight_actions", actions)

    next_obs, rewards, dones, infos = env.step(actions)
    _maybe_sync_cuda("after rollout preflight env.step", device)
    next_privileged_obs = _extract_privileged_obs(infos)
    _log_tensor_summary("next_policy_obs", next_obs)
    _log_tensor_summary("next_critic_obs", next_privileged_obs)
    _log_tensor_summary("rewards", rewards if isinstance(rewards, torch.Tensor) else None)
    _log_tensor_summary("dones", dones if isinstance(dones, torch.Tensor) else None)

    with torch.no_grad():
        _ = runner.alg.act(next_obs, next_privileged_obs)
    _maybe_sync_cuda("after rollout preflight second act", device)
    print("[INFO] Single-step rollout preflight passed before learn.")


def _log_motion_storage(env: gym.Env, device: str | torch.device) -> None:
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    library_cpu_bytes = 0
    library_gpu_bytes = 0
    library_devices: set[str] = set()
    motion_tensor_names = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    active_buffer_names = (
        "_joint_pos",
        "_joint_vel",
        "_body_pos_w",
        "_body_quat_w",
        "_body_lin_vel_w",
        "_body_ang_vel_w",
    )

    for motion in motion_cmd.motions:
        for name in motion_tensor_names:
            tensor = getattr(motion, name)
            nbytes = _tensor_nbytes(tensor)
            library_devices.add(str(tensor.device))
            if tensor.device.type == "cuda":
                library_gpu_bytes += nbytes
            else:
                library_cpu_bytes += nbytes

    active_gpu_bytes = sum(_tensor_nbytes(getattr(motion_cmd, name)) for name in active_buffer_names)
    print(
        "[INFO] Motion storage: "
        f"num_motions={motion_cmd.num_motions}, "
        f"library_devices={sorted(library_devices)}, "
        f"library_cpu={_format_gib(library_cpu_bytes)}, "
        f"library_gpu={_format_gib(library_gpu_bytes)}, "
        f"active_gpu_buffers={_format_gib(active_gpu_bytes)}"
    )

    _log_cuda_memory("after motion storage inspection", device)
    _log_system_gpu_memory("after motion storage inspection", device)


def _disable_training_debug_vis(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if os.getenv("WBT_ENABLE_TRAIN_DEBUG_VIS") == "1":
        print("[INFO] Keeping training debug visualization enabled because WBT_ENABLE_TRAIN_DEBUG_VIS=1.")
        return

    disabled: list[str] = []

    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None and getattr(motion_cfg, "debug_vis", False):
        motion_cfg.debug_vis = False
        disabled.append("commands.motion.debug_vis")

    contact_sensor_cfg = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact_sensor_cfg is not None and getattr(contact_sensor_cfg, "debug_vis", False):
        contact_sensor_cfg.debug_vis = False
        disabled.append("scene.contact_forces.debug_vis")

    if disabled:
        print(f"[INFO] Disabled training debug visualization: {', '.join(disabled)}")


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
    _log_python_env_stack()
    _log_torch_cuda_stack(env_cfg.sim.device)
    _check_blackwell_torch_compatibility(env_cfg.sim.device)
    _run_cublas_smoke_test("before env creation", env_cfg.sim.device)

    # load one or more motion files from wandb registry or local recursive directory
    motion_file, registry_names = _resolve_motion_files()
    env_cfg.commands.motion.motion_file = motion_file
    _disable_training_debug_vis(env_cfg)
    if isinstance(motion_file, list):
        print(f"[INFO] Loaded {len(motion_file)} motion files for training.")
        if args_cli.num_envs is None:
            print(
                "[WARN] --num_envs was not provided, so the task default "
                f"num_envs={env_cfg.scene.num_envs} will be used. "
                "Multi-motion training now keeps reference motions on CPU by default, "
                "but the simulator itself can still exhaust GPU memory on smaller cards. "
                "If you still see CUDA OOM or CUBLAS initialization failures on the target machine, "
                "retry with a smaller --num_envs."
            )

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
    _maybe_sync_cuda("after env creation", agent_cfg.device)
    _run_cublas_smoke_test("after env creation", agent_cfg.device)
    _log_motion_storage(env, agent_cfg.device)
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_names
    )
    _maybe_sync_cuda("after runner init", agent_cfg.device)
    _run_cublas_smoke_test("after runner init", agent_cfg.device)
    _log_cuda_memory("after runner init", agent_cfg.device)
    _log_system_gpu_memory("after runner init", agent_cfg.device)
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
    _preflight_env_and_policy(env, runner, agent_cfg.device)
    _maybe_run_single_step_preflight(env, runner, agent_cfg.device)
    _maybe_sync_cuda("before learn", agent_cfg.device)
    _run_cublas_smoke_test("before learn", agent_cfg.device, batch_size=max(1, env.unwrapped.num_envs))
    _log_cuda_memory("before learn", agent_cfg.device)
    _log_system_gpu_memory("before learn", agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
