import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def sync(device: torch.device, stage: str) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(device_index)
    print(f"[INFO] CUDA synchronize passed @ {stage}")


def print_loaded_cuda_libs() -> None:
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return

    keywords = ("libcublas", "libcudart", "libcudnn", "libcuda", "libcusparse", "libnvrtc")
    libs: set[str] = set()
    for line in maps_path.read_text().splitlines():
        if not any(keyword in line for keyword in keywords):
            continue
        path = line.split()[-1]
        if "/" in path:
            libs.add(path)

    if not libs:
        print("[INFO] Loaded CUDA libs: none found in /proc/self/maps")
        return

    print("[INFO] Loaded CUDA libs:")
    for lib in sorted(libs):
        print(f"[INFO]   {lib}")


def run_linear_backward(device: torch.device, batch_size: int, in_dim: int, out_dim: int) -> None:
    x = torch.randn(batch_size, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(out_dim, device=device, dtype=torch.float32, requires_grad=True)
    y = F.linear(x, weight, bias)
    loss = y.square().mean()
    loss.backward()
    sync(device, f"linear backward bs={batch_size} in={in_dim} out={out_dim}")
    print(f"[INFO] Linear backward passed: batch_size={batch_size}, in_dim={in_dim}, out_dim={out_dim}")


def run_mlp_backward(device: torch.device, batch_size: int, in_dim: int, hidden_dims: list[int], out_dim: int) -> None:
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    model = nn.Sequential(*layers).to(device=device, dtype=torch.float32)

    x = torch.randn(batch_size, in_dim, device=device, dtype=torch.float32)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    sync(device, f"mlp backward bs={batch_size} in={in_dim} out={out_dim}")
    print(
        f"[INFO] MLP backward passed: batch_size={batch_size}, "
        f"shape={in_dim}->{hidden_dims}->{out_dim}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone torch backward smoke test for CUDA/cuBLAS.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 6, 24])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] sys.executable={sys.executable}")
    print(
        "[INFO] Python env stack: "
        f"CONDA_DEFAULT_ENV={os.getenv('CONDA_DEFAULT_ENV')!r}, "
        f"CONDA_PREFIX={os.getenv('CONDA_PREFIX')!r}, "
        f"CONDA_SHLVL={os.getenv('CONDA_SHLVL')!r}"
    )
    print(
        "[INFO] Runtime env: "
        f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')!r}, "
        f"CUDA_HOME={os.getenv('CUDA_HOME')!r}, "
        f"CUDA_PATH={os.getenv('CUDA_PATH')!r}"
    )
    print(f"[INFO] Torch stack: torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(
            "[INFO] CUDA device: "
            f"name={torch.cuda.get_device_name(device_index)}, "
            f"capability={torch.cuda.get_device_capability(device_index)}, "
            f"arch_list={torch.cuda.get_arch_list()}"
        )
    print(
        "[INFO] Backend flags: "
        f"allow_tf32_matmul={torch.backends.cuda.matmul.allow_tf32}, "
        f"allow_tf32_cudnn={torch.backends.cudnn.allow_tf32}"
    )
    print_loaded_cuda_libs()

    for batch_size in args.batch_sizes:
        run_linear_backward(device, batch_size, 160, 512)
        run_linear_backward(device, batch_size, 286, 512)
        run_mlp_backward(device, batch_size, 160, [512, 256, 128], 29)
        run_mlp_backward(device, batch_size, 286, [512, 256, 128], 1)

    print("[INFO] All standalone backward smoke tests passed.")


if __name__ == "__main__":
    main()
