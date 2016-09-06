#!/bin/bash

set -o nounset
set -o errexit


echo "BEGIN: $(date +'%D %T')"
START_TIME=$(date +'%s')
TIME_STAMP=$(date +'%m-%d-%H-%M')

echo "STATEMENT=${STATEMENT}"
#
# main content
#

# change working directory to src
project_dir="/home/jxc761/depth"
src_dir="${project_dir}/src"
cd "${src_dir}" 

# Load troch7 module
module load torch
module load gnuplot
module load cuda/7.0.28

# run torch
th ${STATEMENT} 

module unload torch
module unload gunplot
module unload cuda

STOP_TIME=$(date +'%s')
echo "DONE: $(date +'%D %T')"
echo "TIME: $(($STOP_TIME-$START_TIME))"
