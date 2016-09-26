#include <time.h>
#include <stdio.h>


#include "utils.h"
#include "depth_utils.h"
using namespace nplab;


int main(int argc, char** argv) {

	char pzOutput[512];
	sprintf(pzOutput, "%s", argv[1]);
	int w = atoi(argv[2]);
	int h = atoi(argv[3]);


	printf("pzOutput=%s\r\n", pzOutput);
	printf("w=%d\r\n", w);
	printf("h=%d\r\n", h);

	int offset = 0;
	int numb   = 500;

	// start timer 
	clock_t start= clock();
  	
  	// process scene by scene
  	enter_embree();

  	// open file for cache data
	FILE * pfile = fopen(pzOutput, "wb");
	if(pfile == 0){
		Error("Cannot open file: %s\r\n", pzOutput);
	}

	for (int s = 0; s < numb; s++) {
		printf("processing scene %d......\r\n",  s+offset);
		// process_scene(s+offset);
		cache_scene(s+offset, w, h, pfile);
	}

	// colse file 
	fclose(pfile);

 	exit_embree();
	
	// stop timer
	clock_t finish=clock();
	double total_time  = 1.0 *(finish-start) / CLOCKS_PER_SEC;
	printf("Speed:%fmin/scene\r\nTotal time: %fmin\r\nDone\r\n", total_time/60/numb, total_time/60);
}



