#!/bin/bash
#
# testing on small dataset 
# vary the number of units of model 
#
set -o nounset
set -o errexit


## who am i? ##
_script="$(readlink -f ${BASH_SOURCE[0]})"
 
## Delete last component from $_script ##
base="$(dirname $_script)"
	

# coarse tune 
alpha=(1.2 2.4 3.6)
for a in "${alpha[@]}" do
  ${base}/tune.sh 'small' 'sym' 3 $a 1e-4 1e-1 4 'log'
done
