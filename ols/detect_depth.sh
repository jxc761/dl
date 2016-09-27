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

DN_CACHE="${PROJECT_DIR}/buffer/cache"
DN_BIN="${PROJECT_DIR}/ols/bin"

RES=16
FN_NRM="${DN_CACHE}/depth_normal_${RES}x${RES}.cache"
FN_TXT="${DN_CACHE}/depth_normal_${RES}x${RES}_undefined.txt"
${DN_BIN}/detect ${FN_NRM} ${FN_TXT} ${RES} ${RES}


RES=32
FN_NRM="${DN_CACHE}/depth_normal_${RES}x${RES}.cache"
FN_TXT="${DN_CACHE}/depth_normal_${RES}x${RES}_undefined.txt"
${DN_BIN}/detect ${FN_NRM} ${FN_TXT} ${RES} ${RES}