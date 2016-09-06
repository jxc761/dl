#!/bin/bash
# usage: 
#   test_navie.sh 
#   test_navie.sh idx
#   test_navie.sh first last
  
set -o nounset
set -o errexit



# parse argument
FIRST=${1:-1}
LAST=6
if [ $# -eq 1 ]; then
	LAST=FIRST
elif [ $# -eq 2 ]; then
	LAST=$2
else
	echo "wrong number of arguments"
	exit
fi

 
echo "FIRST=$FIRST"
echo "LAST=$LAST"
script_dir='~/depth/script'
for i in $(seq $FIRST $LAST); do 
	${script_dir}/run.sh "naive_learning_rate$i" "test_naive.lua $i" 
done

