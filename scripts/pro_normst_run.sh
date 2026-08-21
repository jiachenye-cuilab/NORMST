#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 --round-id ID --round-reason TEXT --manifest PATH --output-dir PATH --variant VARIANT --seed SEED --physical-gpu INDEX [--candidate-lock PATH] [--idw-cache-dir PATH] [--mode train|resume|predict] [--smoke] [--dry-run]"
}

ROUND_ID=""
ROUND_REASON=""
RUN_MANIFEST=""
OUTPUT_DIR=""
VARIANT=""
SEED=""
PHYSICAL_GPU=""
CANDIDATE_LOCK=""
IDW_CACHE_DIR=""
MODE="train"
SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --round-id) ROUND_ID="$2"; shift 2 ;;
    --round-reason) ROUND_REASON="$2"; shift 2 ;;
    --manifest) RUN_MANIFEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --physical-gpu) PHYSICAL_GPU="$2"; shift 2 ;;
    --candidate-lock) CANDIDATE_LOCK="$2"; shift 2 ;;
    --idw-cache-dir) IDW_CACHE_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

for value in ROUND_ID ROUND_REASON RUN_MANIFEST OUTPUT_DIR VARIANT SEED PHYSICAL_GPU; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required setting: ${value}"
    usage
    exit 2
  fi
done
if [[ ! "${ROUND_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "Invalid round identity: ${ROUND_ID}"
  exit 2
fi
if [[ ! "${VARIANT}" =~ ^(full|one-shot|local-only|global-only)$ ]]; then
  echo "Invalid variant: ${VARIANT}"
  exit 2
fi
if [[ ! "${MODE}" =~ ^(train|resume|predict)$ ]]; then
  echo "Invalid mode: ${MODE}"
  exit 2
fi
if [[ ! -r "${RUN_MANIFEST}" ]]; then
  echo "Manifest is not readable: ${RUN_MANIFEST}"
  exit 2
fi
if [[ "${MODE}" == "train" && -e "${OUTPUT_DIR}" ]]; then
  echo "Fresh output directory already exists: ${OUTPUT_DIR}"
  exit 2
fi
if [[ "${MODE}" != "train" && ! -d "${OUTPUT_DIR}" ]]; then
  echo "Existing run directory is required for mode ${MODE}: ${OUTPUT_DIR}"
  exit 2
fi

EXPECTED_PYTHON="/data/yejiachen/Software/miniconda3/envs/NORMST/bin/python"
ACTUAL_PYTHON="$(conda run -n NORMST python -c 'import sys; print(sys.executable)')"
if [[ "${ACTUAL_PYTHON}" != "${EXPECTED_PYTHON}" ]]; then
  echo "Unexpected NORMST Python: ${ACTUAL_PYTHON}"
  exit 1
fi
conda run -n NORMST python -c 'import os, scanpy; required=("NUMBA_CACHE_DIR","MPLCONFIGDIR","XDG_CACHE_HOME"); missing=[key for key in required if not os.environ.get(key)]; assert not missing, f"missing cache settings: {missing}"'

GPU_STATE="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)"
IFS=',' read -r GPU_INDEX GPU_NAME GPU_TOTAL GPU_USED GPU_FREE GPU_UTIL <<< "${GPU_STATE}"
GPU_FREE="$(xargs <<< "${GPU_FREE}")"
GPU_UTIL="$(xargs <<< "${GPU_UTIL}")"
if (( GPU_FREE < 60000 || GPU_UTIL > 20 )); then
  echo "GPU ${PHYSICAL_GPU} does not have the required safety margin: ${GPU_STATE}"
  exit 1
fi

CONTROL_ROOT="${OUTPUT_DIR}.control"
mkdir -p "${CONTROL_ROOT}"
CONTROL_DIR="$(mktemp -d "${CONTROL_ROOT}/${MODE}.XXXXXX")"
LOG_PATH="${CONTROL_DIR}/train.log"
PREFLIGHT_LOG="${CONTROL_DIR}/contract_tests.log"
COMMAND_PATH="${CONTROL_DIR}/command.txt"
PREFLIGHT_COMMAND_PATH="${CONTROL_DIR}/contract_tests_command.txt"
METADATA_PATH="${CONTROL_DIR}/launcher.json"

COMMAND=(
  conda run --no-capture-output -n NORMST python -u train.py
  --task visium
  --model pro-normst
  --manifest "${RUN_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --variant "${VARIANT}"
  --seed "${SEED}"
  --round-id "${ROUND_ID}"
  --round-reason "${ROUND_REASON}"
  --device cuda:0
)
if [[ -n "${CANDIDATE_LOCK}" ]]; then
  COMMAND+=(--candidate-lock "${CANDIDATE_LOCK}")
fi
if [[ -n "${IDW_CACHE_DIR}" ]]; then
  COMMAND+=(--idw-cache-dir "${IDW_CACHE_DIR}")
fi
if [[ "${MODE}" == "resume" ]]; then
  COMMAND+=(--resume)
elif [[ "${MODE}" == "predict" ]]; then
  COMMAND+=(--predict-only)
fi
if (( SMOKE == 1 )); then
  COMMAND+=(--smoke)
fi

printf 'CUDA_VISIBLE_DEVICES=%q ' "${PHYSICAL_GPU}" > "${COMMAND_PATH}"
printf '%q ' "${COMMAND[@]}" >> "${COMMAND_PATH}"
printf '\n' >> "${COMMAND_PATH}"
printf 'CUDA_VISIBLE_DEVICES=%q conda run --no-capture-output -n NORMST python -m unittest discover -s tests -v\n' "${PHYSICAL_GPU}" > "${PREFLIGHT_COMMAND_PATH}"
jq -n \
  --arg round_identity "${ROUND_ID}" \
  --arg round_reason "${ROUND_REASON}" \
  --arg manifest "$(realpath "${RUN_MANIFEST}")" \
  --arg output_dir "$(realpath -m "${OUTPUT_DIR}")" \
  --arg variant "${VARIANT}" \
  --argjson seed "${SEED}" \
  --argjson physical_gpu "${PHYSICAL_GPU}" \
  --arg gpu_state "${GPU_STATE}" \
  --arg mode "${MODE}" \
  --arg idw_cache_dir "${IDW_CACHE_DIR}" \
  --arg log "${LOG_PATH}" \
  --arg contract_test_log "${PREFLIGHT_LOG}" \
  '{schema:"pro-normst-launch-v1",status:"preflight",round_identity:$round_identity,round_reason:$round_reason,manifest:$manifest,output_dir:$output_dir,variant:$variant,seed:$seed,physical_gpu:$physical_gpu,gpu_state:$gpu_state,mode:$mode,idw_cache_dir:(if $idw_cache_dir == "" then null else $idw_cache_dir end),log:$log,contract_test_log:$contract_test_log}' \
  > "${METADATA_PATH}"

set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run --no-capture-output -n NORMST \
  python -m unittest discover -s tests -v > "${PREFLIGHT_LOG}" 2>&1
PREFLIGHT_STATUS=$?
set -e
if (( PREFLIGHT_STATUS != 0 )); then
  jq --argjson exit_code "${PREFLIGHT_STATUS}" '.status="preflight-failed" | .preflight_exit_code=$exit_code' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
  mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
  exit "${PREFLIGHT_STATUS}"
fi
PREFLIGHT_SHA256="$(sha256sum "${PREFLIGHT_LOG}" | cut -d ' ' -f 1)"
jq --arg sha256 "${PREFLIGHT_SHA256}" '.preflight_exit_code=0 | .contract_test_log_sha256=$sha256' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"

if (( DRY_RUN == 1 )); then
  jq '.status="dry-run"' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
  mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
  echo "Dry-run preflight complete: ${CONTROL_DIR}"
  exit 0
fi

set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${COMMAND[@]}" > "${LOG_PATH}" 2>&1 &
RUN_PID=$!
jq --argjson pid "${RUN_PID}" '.status="running" | .pid=$pid' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
wait "${RUN_PID}"
RUN_STATUS=$?
set -e
jq --argjson exit_code "${RUN_STATUS}" '.status=(if $exit_code == 0 then "complete" else "failed" end) | .exit_code=$exit_code' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
exit "${RUN_STATUS}"
