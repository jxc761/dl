#!/bin/bash
# 
# Usage: 
#   runexp.sh
#   runexp.sh index1 ...

set -o nounset
set -o errexit

# parse arguments
IDX=$(seq 1 10)
if [ "$#" -gt 0 ]; then
	IDX=($@)  #"$@"
fi


SCRIPT="${HOME}/dl/script/run.sh"
for i in "${IDX[@]}"; do
	JOBNAME="depthExp01_runexp01_$i"
	${SCRIPT} "${JOBNAME}" "depthExp01" "exp.lua conf01.lua $i" "gpu"
done