#!/bin/bash
# usage: 
#   test_navie.sh 
#   test_navie.sh idx
#   test_navie.sh first last

set -o nounset
set -o errexit

if [ $# -gt 2 ]; then 
	echo "wrong number of arguments"
	exit
fi

# parse arguments
FIRST=${1:-1}
LAST=6
if [ $# -eq 1 ]; then
	LAST=FIRST
elif [ $# -eq 2 ]; then
	LAST=$2
fi
 
echo "FIRST=$FIRST"
echo "LAST=$LAST"
script_dir="${HOME}/depth/script"
for i in $(seq $FIRST $LAST); do 
	${script_dir}/run.sh "naive_gray_inverse_coarse_learningrate$i" "test_naive.lua coarse_tuning $i" "gpu" 
done



