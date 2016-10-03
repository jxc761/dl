#!/bin/bash
# USAGE: run.sh <JOB_NAME> <LUA_SCIRPT_FILENAME> <MODE> <MEMORY>
# 
#

# print usage
if [ "$#" -lt 3 ]; then
  echo "Usage: "
  echo "run.sh.sh <JOB_NAME> <WORK_DIR> <LUA_SCIRPT_FILENAME> [<MODE> [<MEMORY>]]"
  echo "  <WORK_DIR>: relative path to project_dir"
  echo "  <LUA_SCIRPT_FILENAME>: relative path to work_dir"
  echo "  <MODE>  : {cpu}   | gpu"
  echo "  <MEMORY>: {small} | large"
  exit
fi


set -o nounset
set -o errexit

PROJECT_DIR="/home/jxc761/dl"
SCRIPT_DIR="${PROJECT_DIR}/script"
LOG_DIR="${PROJECT_DIR}/log/$2"

JOB_NAME=$1
WORK_DIR="${PROJECT_DIR}/$2"
STATEMENT=$3
MODE=${4:-cpu}  # cpu | gpu
MEMORY=${5:-small} # large | small 
 


FN_OUT="${LOG_DIR}/${JOB_NAME}.out"
FN_ERR="${LOG_DIR}/${JOB_NAME}.err"
test -d "${LOG_DIR}" || mkdir -p "${LOG_DIR}"


[[ ${MEMORY} == small ]] &&  mem="--mem=20g" || mem="-mem=40g"
[[ ${MODE} == cpu ]]     &&  partition="--partition=batch" || partition="--partition=gpufermi"
[[ ${MODE} == cpu ]]     &&  gres="" || gres="--gres=gpu:1"

sbatch --job-name="${JOB_NAME}"  --output="${FN_OUT}" --error="${FN_ERR}"\
	 --nodes=1 --cpus-per-task=12 "${mem}" "${partition}"  "${gres}" \
	 --workdir="${WORK_DIR}" \
	 --export=STATEMENT="${STATEMENT}" \
     "${SCRIPT_DIR}/run_lua.sh"						



