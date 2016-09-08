#!/bin/bash
# usage: 
#   ./tune_learn_rate.sh [min max [n [method]]]
#
# examples:
#   ./tune_learn_rate.sh
#   ./tune_learn_rate.sh 0.005 0.5 12 
#   ./tune_learn_rate.sh 0.005 0.5 12 linear
#

set -o nounset
set -o errexit


if [ $# -gt 4 ]; then 
	echo "wrong number of arguments"
	exit
fi

# parse arguments
min=${1:-1e-6}
max=${2:-1e-1}
n=${3:-6}
method=${4:-log}


echo "min=${min}"
echo "max=${max}"
echo "n=${n}"
echo "method=${method}"


script_dir="${HOME}/depth/script"
jobname="tuning_learning_rate_${min}_${max}_${n}_${method}"

${script_dir}/run.sh "${jobname}" "exp01" "tune_learning_rate.lua $min $max $n $method" "gpu" 



