#!/bin/bash
# usage: 
#   test_use_gpu.sh
#
set -o nounset
set -o errexit

script_dir="${HOME}/depth/script"

${script_dir}/run.sh "run" "exp01" "run.lua" "gpu"
