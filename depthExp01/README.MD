# Goal 

Mapping the multiple images to a single depthmap.

# Experiments
16x16|32x32 x 2|3  fc|conv 16x16|32x32 x1(inverse|log|inverse)


#results


实验设计
输出：
./buffer/depthExp01/step1
	-01
	  -<learning_rate>/
	  	- process.txt
	  	- exp_process.txt
	  	- exp_process.dat
	  	- performance.txt
	  	- exp_hist.txt
	  	- exp_eval.dat
	  -<learning_rate>
	-02
	-03
	...
	-40

exp_process.dat
	- expX, expY, predictY1, predictY2, ..., predictYn
exp_eval.dat
	- expX 