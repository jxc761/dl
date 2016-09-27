#!/bin/bash
#SBATCH --mem=10g
#SBATCH --cpus-per-task=20
#

set -o nounset

#echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
#echo "SLURM_SUBMIT_HOST=${SLURM_SUBMIT_HOST}"
#echo "SLURMD_NODENAME=${SLURMD_NODENAME}"
#echo "SLURM_TASK_PID=${SLURM_TASK_PID}"
#echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"

PROJECT_DIR="${HOME}/dl"

if [ $(uname) == "Darwin" ]; then
    PROJECT_DIR="${HOME}/Dropbox/dev/dl"
fi
echo ${PROJECT_DIR}

DN_CACHE="${PROJECT_DIR}/buffer/cache"
WORK_DIR="${PROJECT_DIR}/ols/image"
COMMAND="cache_image('${DN_CACHE}/image_gray_16x16.cache', 16)"

cd "${WORK_DIR}"

echo "BEGIN: $(date +'%D, %T')"
START_TIME=$(date +'%s')


module load matlab

# scripts for while using versions R2014b/R2015b,  
# MATLAB Preference Setting
matlab_prefdir="/tmp/jxc761/matlab/`hostname`_PID$$"
test -d $matlab_prefdir || mkdir -p $matlab_prefdir
export MATLAB_PREFDIR="$matlab_prefdir"

matlab -nodisplay  -r "${COMMAND}"  # >  "/home/jxc761/benchmarks/logs/job_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

module unload matlab

STOP_TIME=$(date +'%s')
echo "DONE: $(date +'%D, %T')"
echo "TIME: $(($STOP_TIME-$START_TIME))"