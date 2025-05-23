#!/usr/bin/env bash

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"
echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

ROOT="checkpoints"
ORIGINAL="ibm-granite/granite-3.3-8b-instruct"
TRAINED="${ROOT}/granite-3.3-8b-instruct-teigaku-genzei"
INTERPOLATED="${TRAINED}-interp"

MODEL_PATH="${ORIGINAL}"
TRAINED_MODEL_PATH="${TRAINED}"
TRAINED_WEIGHT=0.5
INTERPOLATED_MODEL_PATH="${INTERPOLATED}"
TORCH_DTYPE="bfloat16"

ENV=""
#ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
#ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"
cmd="${ENV}python ${BASENAME}.py --model_path ${MODEL_PATH} --trained_model_path ${TRAINED_MODEL_PATH} --trained_weight ${TRAINED_WEIGHT} --interpolated_model_path ${INTERPOLATED_MODEL_PATH} --torch_dtype ${TORCH_DTYPE}"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
