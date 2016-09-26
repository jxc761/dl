/***
 * Usage: detect <path_to_output> <path_to_input> w h
 * 
 * Example:
 *   
 *   detect depth_16x16_undefinded_scenes.txt depth_normal_16x16.cache 16 16
 */
#include <stdio.h>
#include <stdlib.h>



bool hasBadPixels(float* depth, int n) {
	for (int i = 0; i < n; i++) {
		if(depth[i] < 0)
			return true;
	}
	return false;
}


int main(int argc, char** argv){
	char pzInput[512];
	char pzOutput[512];
	int  w=0, h=0;

	sprintf(pzOutput, "%s", argv[1]);
	sprintf(pzInput, "%s", argv[2]);
	w = atoi(argv[3]);
	h = atoi(argv[4]);

	printf("pzOutput=%s\r\n", pzOutput);
	printf("pzInput=%s\r\n", pzInput);
	printf("w=%d\r\n", w);
	printf("h=%d\r\n", h);

	int S = 500;
	int T = 40;
	int F = 30;
	int n = w * h;

	FILE* fin = fopen(pzInput, "rb");
	if (fin==0) {
		printf("Cannot open file: %s!\r\n", pzInput);
		return -1;
	}

	FILE* fout = fopen(pzOutput, "w");
	if (fout==0) {
		printf("Cannot open file: %s!\r\n", pzOutput);
		return -2;
	}


	float* depth = new float[n];

	for (int i=0;i<S; i++){
		printf("processing scene %d \r\n", i);
		for (int j=0; j<T; j++) {
			for(int k=0; k<F; k++){
				fread(depth,  sizeof(float), n, fin);
				if(hasBadPixels(depth, n)) {
					fprintf(fout, "%d\t%d\t%d\r\n", i, j, k);
				}
			}// end for k
		}// end for j
		fflush(fout);
	}
	delete[] depth;


	fclose(fout);
	fclose(fin);

}