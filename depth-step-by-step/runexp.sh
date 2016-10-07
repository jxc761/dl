#!/bin/bash
# 
# Usage: 
#   runexp.sh
#   runexp.sh index1 ...

set -o nounset
set -o errexit



function run(){
 
 MIN=$1
 MAX=$2
 N=$3
 METHOD=$4

 SCRIPT="${HOME}/dl/script/run.sh"
 JOBNAME="depth_step_by_step_$MIN_$MAX_$N_$METHOD"
 STATEMENT="exp01.lua $MIN $MAX $N $METHOD"
 WORKDIR="depth-step-by-step"

 ${SCRIPT} "${JOBNAME}" "${WORKDIR}" "${STATEMENT}" "gpu"
}


run 0.5 3 4 linear


#SCRIPT="${HOME}/dl/script/run.sh"

#JOBNAME="depthExp01_d01_m01_tr01"
#${SCRIPT} "${JOBNAME}" "depthExp01" "exp01.lua" "gpu"


# JOBNAME="depthExp01_d01_m01_tr02"
# ${SCRIPT} "${JOBNAME}" "depthExp01" "exp02.lua" "gpu"

# JOBNAME="depthExp01_d01_m01_tr03"
# ${SCRIPT} "${JOBNAME}" "depthExp01" "exp03.lua" "gpu"

# parse arguments
# IDX=$(seq 1 10)
# if [ "$#" -gt 0 ]; then
#	IDX=($@)  #"$@"
# fi
