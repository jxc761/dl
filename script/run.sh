#!/bin/bash
# USAGE: run.sh <JOB_NAME> <LUA_SCIRPT_FILENAME> <MODE> <MEMORY>
# 
#

# print usage
if [ "$#" -lt 2 ]; then
  echo "Usage: "
  echo "run.sh.sh <JOB_NAME> <LUA_SCIRPT_FILENAME> [<MODE> [<MEMORY>]]"
  echo "  <LUA_SCIRPT_FILENAME>: relative path to project_dir/src"
  echo "  <MODE>  : {cpu}   | gpu"
  echo "  <MEMORY>: {small} | large"
  exit
fi


set -o nounset
set -o errexit

JOB_NAME=$1
STATEMENT=$2
MODE=${3:-cpu}  # cpu | gpu
MEMORY=${4:-small} # large | small 
 
PROJECT_DIR="/home/jxc761/depth"
SCRIPT_DIR="${PROJECT_DIR}/script"
LOG_DIR="${PROJECT_DIR}/log"
FN_OUT="${LOG_DIR}/${JOB_NAME}.out";
test -d "${LOG_DIR}" || mkdir -p "${LOG_DIR}"


[[ ${MEMORY} == small ]] &&  mem="--mem=20g" || mem="-mem=40g"
[[ ${MODE} == cpu ]]     &&  partition="--partition=batch" || partition="-p gpufermi --gres=gpu:1"

sbatch --job-name="${JOB_NAME}"  --output="${FN_OUT}" \
	 --nodes=1 --cpus-per-task=12 "${mem}" "${partition}"   \
	 --export=STATEMENT="${STATEMENT}" \
     "${SCRIPT_DIR}/run_lua.sh"						



