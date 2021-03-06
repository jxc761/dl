#!/bin/bash
# 
# Usage: 
#   runexp.sh
#   runexp.sh index1 ...

set -o nounset
set -o errexit


SCRIPT="${HOME}/dl/script/run.sh"

JOBNAME="depthExp01_d01_m01_tr01"
${SCRIPT} "${JOBNAME}" "depthExp01" "exp01.lua" "gpu"


# JOBNAME="depthExp01_d01_m01_tr02"
# ${SCRIPT} "${JOBNAME}" "depthExp01" "exp02.lua" "gpu"

# JOBNAME="depthExp01_d01_m01_tr03"
# ${SCRIPT} "${JOBNAME}" "depthExp01" "exp03.lua" "gpu"

# parse arguments
# IDX=$(seq 1 10)
# if [ "$#" -gt 0 ]; then
#	IDX=($@)  #"$@"
# fi
