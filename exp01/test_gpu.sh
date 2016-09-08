#!/bin/bash
# usage: 
#   test_use_gpu.sh
#
set -o nounset
set -o errexit

script_dir="${HOME}/depth/script"

${script_dir}/run.sh "test_gpu_0" "exp01" "test_gpu.lua 0" "gpu"
${script_dir}/run.sh "test_gpu_1" "exp01" "test_gpu.lua 1" "gpu" 



