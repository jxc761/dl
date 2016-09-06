#!/bin/bash
# usage: 
#   fine_tuning_naive.sh min max [n]
#
set -o nounset
set -o errexit

if [ $# -ne 2 ]; then 
	echo "wrong number of arguments"
	exit
fi

# parse arguments
min=$1
max=$2
n=${3:-6}

echo "min=${min}"
echo "max=${max}"
echo "n=${n}"

script_dir="${HOME}/depth/script"
for i in $(seq 1 $n); do 
	${script_dir}/run.sh "naive_gray_inverse_fine_learningrate$i" "test_naive.lua fine_tuning ${min} ${max} ${n} $i" 
done



