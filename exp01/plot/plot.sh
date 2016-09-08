#!/bin/bash

gnuplot plot_peformances1.gpl
gnuplot plot_peformances2.gpl


input_dir="/Users/Jing/Dropbox/dev/depth/buffer/exp01/tune_lr_0.005000_0.500000_12_log"
fn_output="/Users/Jing/Dropbox/dev/depth/buffer/exp01/tune_lr_0.005000_0.500000_12_log/processes.png"
settings="input_dir='${input_dir}';fn_output='${fn_output}';"
gnuplot -e $settings plot_process.gpl


input_dir="/Users/Jing/Dropbox/dev/depth/buffer/exp01/tune_lr_0.000001_0.100000_6_log"
fn_output="/Users/Jing/Dropbox/dev/depth/buffer/exp01/tune_lr_0.000001_0.100000_6_log/processes.png"
settings="input_dir='${input_dir}';fn_output='${fn_output}';"
gnuplot -e $settings plot_process.gpl