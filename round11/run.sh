#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 --manifest PATH --output-dir PATH --physical-gpu INDEX [--smoke]"
}

RUN_MANIFEST=""
OUTPUT_DIR=""
PHYSICAL_GPU=""
SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) RUN_MANIFEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --physical-gpu) PHYSICAL_GPU="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done
for value in RUN_MANIFEST OUTPUT_DIR PHYSICAL_GPU; do
  if [[ -z "${!value}" ]]; then echo "Missing required setting: ${value}"; exit 2; fi
done
if [[ ! -r "${RUN_MANIFEST}" ]]; then echo "Manifest is not readable: ${RUN_MANIFEST}"; exit 2; fi
if [[ -e "${OUTPUT_DIR}" ]]; then echo "Fresh output directory already exists: ${OUTPUT_DIR}"; exit 2; fi

EXPECTED_PYTHON="/data/yejiachen/Software/miniconda3/envs/NORMST/bin/python"
ACTUAL_PYTHON="$(conda run -n NORMST python -c 'import sys; print(sys.executable)')"
if [[ "${ACTUAL_PYTHON}" != "${EXPECTED_PYTHON}" ]]; then echo "Unexpected NORMST Python: ${ACTUAL_PYTHON}"; exit 1; fi
GPU_STATE="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)"
IFS=',' read -r GPU_INDEX GPU_NAME GPU_TOTAL GPU_USED GPU_FREE GPU_UTIL <<< "${GPU_STATE}"
GPU_FREE="$(xargs <<< "${GPU_FREE}")"
GPU_UTIL="$(xargs <<< "${GPU_UTIL}")"
if (( GPU_FREE < 60000 || GPU_UTIL > 20 )); then echo "GPU ${PHYSICAL_GPU} lacks safety margin: ${GPU_STATE}"; exit 1; fi

ROUND_ID="pro-v2-round-011"
ROUND_REASON="decay Pearson warm-start to zero before SmoothL1-only refinement"
CONTROL_ROOT="${OUTPUT_DIR}.control"
mkdir -p "${CONTROL_ROOT}"
CONTROL_DIR="$(mktemp -d "${CONTROL_ROOT}/train.XXXXXX")"
LOG_PATH="${CONTROL_DIR}/train.log"
PREFLIGHT_LOG="${CONTROL_DIR}/contract_tests.log"
COMMAND_PATH="${CONTROL_DIR}/command.txt"
METADATA_PATH="${CONTROL_DIR}/launcher.json"
COMMAND=(conda run --no-capture-output -n NORMST python -u -m round11.train --manifest "${RUN_MANIFEST}" --output-dir "${OUTPUT_DIR}" --variant full --seed 2027 --round-id "${ROUND_ID}" --round-reason "${ROUND_REASON}" --device cuda:0)
if (( SMOKE == 1 )); then COMMAND+=(--smoke); fi
printf 'CUDA_VISIBLE_DEVICES=%q ' "${PHYSICAL_GPU}" > "${COMMAND_PATH}"
printf '%q ' "${COMMAND[@]}" >> "${COMMAND_PATH}"
printf '\n' >> "${COMMAND_PATH}"
jq -n --arg round_identity "${ROUND_ID}" --arg round_reason "${ROUND_REASON}" --arg manifest "$(realpath "${RUN_MANIFEST}")" --arg output_dir "$(realpath -m "${OUTPUT_DIR}")" --argjson physical_gpu "${PHYSICAL_GPU}" --arg gpu_state "${GPU_STATE}" --arg log "${LOG_PATH}" --arg contract_test_log "${PREFLIGHT_LOG}" --argjson smoke "${SMOKE}" '{schema:"pro-normst-round11-launch-v1",status:"preflight",round_identity:$round_identity,round_reason:$round_reason,manifest:$manifest,output_dir:$output_dir,physical_gpu:$physical_gpu,gpu_state:$gpu_state,smoke:($smoke==1),log:$log,contract_test_log:$contract_test_log}' > "${METADATA_PATH}"

set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run --no-capture-output -n NORMST python -m unittest discover -s tests -v > "${PREFLIGHT_LOG}" 2>&1
BASE_STATUS=$?
if (( BASE_STATUS == 0 )); then CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run --no-capture-output -n NORMST python -m unittest discover -s round10/tests -v >> "${PREFLIGHT_LOG}" 2>&1; R10_STATUS=$?; else R10_STATUS=99; fi
if (( BASE_STATUS == 0 && R10_STATUS == 0 )); then CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" conda run --no-capture-output -n NORMST python -m unittest discover -s round11/tests -v >> "${PREFLIGHT_LOG}" 2>&1; R11_STATUS=$?; else R11_STATUS=99; fi
set -e
if (( BASE_STATUS != 0 || R10_STATUS != 0 || R11_STATUS != 0 )); then
  jq --argjson base_exit "${BASE_STATUS}" --argjson r10_exit "${R10_STATUS}" --argjson r11_exit "${R11_STATUS}" '.status="preflight-failed" | .base_test_exit_code=$base_exit | .round10_test_exit_code=$r10_exit | .round11_test_exit_code=$r11_exit' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
  mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
  exit 1
fi
PREFLIGHT_SHA256="$(sha256sum "${PREFLIGHT_LOG}" | cut -d ' ' -f 1)"
jq --arg sha256 "${PREFLIGHT_SHA256}" '.status="running" | .base_test_exit_code=0 | .round10_test_exit_code=0 | .round11_test_exit_code=0 | .contract_test_log_sha256=$sha256' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${COMMAND[@]}" > "${LOG_PATH}" 2>&1
RUN_STATUS=$?
set -e
jq --argjson exit_code "${RUN_STATUS}" '.status=(if $exit_code == 0 then "complete" else "failed" end) | .exit_code=$exit_code' "${METADATA_PATH}" > "${METADATA_PATH}.tmp"
mv "${METADATA_PATH}.tmp" "${METADATA_PATH}"
exit "${RUN_STATUS}"
