#!/bin/bash
# 
# cache_depth.sh 
#

# the directory 


CUR_DIR="$( cd "$(dirname $0)" && pwd)"
PROJECT_DIR="$(dirname $CUR_DIR)"

# compile the tools
make 


DN_CACHE="${PROJECT_DIR}/buffer/cache"
DN_BIN="${PROJECT_DIR}/ols/bin"
[[ ! -d ${DN_CACHE} ]] && mkdir -p ${DN_CACHE}

function cache(){
	RES=$1

	FN_NRM="${DN_CACHE}/depth_noraml_${RES}x${RES}.cache"
	FN_INV="${DN_CACHE}/depth_inverse_${RES}x${RES}.cache"
	FN_LOG="${DN_CACHE}/depth_log_${RES}x${RES}.cache"

	if [[ ! -f ${FN_NRM} ]]; then
		${DN_BIN}/cache_depth ${FN_NRM} ${RES} ${RES} 
	fi

	if [[ ! -f ${FN_INV} ]]; then
		${DN_BIN}/convert_depth ${FN_NRM} normal ${FN_INV} inverse ${RES} ${RES} 
	fi

	if [[ ! -f ${FN_LOG} ]]; then
		${DN_BIN}/convert_depth ${FN_NRM} normal ${FN_LOG} log ${RES} ${RES} 
	fi

}

cache 16
cache 32
