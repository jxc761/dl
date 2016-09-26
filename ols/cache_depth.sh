#!/bin/bash
# 
# cache_depth.sh 
#

# the directory 

PROJECT_DIR="${HOME}/dl"

if [ $(uname) == "Darwin" ]; then
    PROJECT_DIR="${HOME}/Dropbox/dev/dl"
fi
echo ${PROJECT_DIR}

# compile the tools
make 


DN_CACHE="${PROJECT_DIR}/buffer/cache"
DN_BIN="${PROJECT_DIR}/ols/bin"
DN_LOG="${PROJECT_DIR}/ols/logs"

[[ ! -d ${DN_CACHE} ]] && mkdir -p ${DN_CACHE}
[[ ! -d ${DN_LOG} ]]  && mkdir -p ${DN_LOG}
function cache(){
	RES=$1

	FN_NRM="${DN_CACHE}/depth_normal_${RES}x${RES}.cache"
	FN_INV="${DN_CACHE}/depth_inverse_${RES}x${RES}.cache"
	FN_LOG="${DN_CACHE}/depth_log_${RES}x${RES}.cache"

	if [[ ! -f ${FN_NRM} ]]; then
		${DN_BIN}/cache_depth ${FN_NRM} ${RES} ${RES} > "${DN_LOG}/depth_normal_${RES}x${RES}.log" 
	fi

	if [[ ! -f ${FN_INV} ]]; then
		${DN_BIN}/convert_depth ${FN_NRM} normal ${FN_INV} inverse ${RES} ${RES} > "${DN_LOG}/depth_inverse_${RES}x${RES}.log" 
	fi

	if [[ ! -f ${FN_LOG} ]]; then
		${DN_BIN}/convert_depth ${FN_NRM} normal ${FN_LOG} log ${RES} ${RES} > "${DN_LOG}/depth_log_${RES}x${RES}.log" 
	fi

}

cache 16
cache 32
