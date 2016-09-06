#!/bin/bash
# usage: 
#   test_use_gpu.sh
#
set -o nounset
set -o errexit

script_dir="${HOME}/depth/script"
${script_dir}/run.sh "naive_gray_inverse_testgpu" "test_naive.lua test_gpu 0" "gpu"
${script_dir}/run.sh "naive_gray_inverse_testgpu" "test_naive.lua test_gpu 1" "gpu" 



