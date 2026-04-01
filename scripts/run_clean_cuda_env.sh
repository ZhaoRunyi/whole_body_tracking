#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]"
  exit 1
fi

filter_ld_library_path() {
  local input="${1:-}"
  local -a kept=()
  local IFS=':'
  read -r -a parts <<< "$input"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    case "$part" in
      /usr/local/cuda-*|/usr/local/cuda-*/lib64|/home/*/TensorRT/lib|/opt/onnxruntime/lib)
        continue
        ;;
    esac
    kept+=("$part")
  done
  (IFS=':'; echo "${kept[*]}")
}

export LD_LIBRARY_PATH="$(filter_ld_library_path "${LD_LIBRARY_PATH:-}")"
echo "[INFO] Sanitized LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
exec "$@"
