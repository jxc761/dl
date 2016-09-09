#!/bin/bash
set -o nounset
set -o errexit

# unpack arguments
dataset=$1

struct=$2
nhidden=$3
alpha=$4

min=$5
max=$6
n=$7
method=$8

# print out arguments
echo "----------------------------------------------------"
echo "data:"
echo "dataset=${dataset}"
echo "model:"
echo "struct=${struct}"
echo "nhidden=${nhidden}"
echo "alpha=${alpha}"
echo "optim:"
echo "min=${min}"
echo "max=${max}"
echo "n=${n}"
echo "method=${method}"
echo "----------------------------------------------------"


# main content
script_dir="${HOME}/depth/script"
jobname="${dataset}_${struct}_${nhidden}_${alpha}_${min}_${max}_${n}_${method}"
${script_dir}/run.sh "${jobname}" "exp02" "tune.lua ${dataset} ${struct} ${nhidden} ${alpha} ${min} ${max} ${n} ${method}" "gpu" 
