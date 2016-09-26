#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define INF (1e15)

void nrm2log(const float* input, float* output, int n) {
	for (int i=0; i<n;i++){
		if (input[i] < 0 ) {
			output[i] = INF;
		} else {
			output[i] = log(input[i]);
		}
	}
}

void nrm2inv(const float* input, float* output, int n) {
	for (int i=0; i<n; i++){
		if (input[i] < 0 ) {
			output[i] = 0;
		} else {
			output[i] = 1.0 / input[i];
		}
	}
}

int main(int argc, char** argv){
	char pzInput[512];
	char spacex[10];
	char pzOutput[512];
	char spacey[10];
	int  w=0, h=0;

	sprintf(pzInput, "%s", argv[1]);
	sprintf(spacex, "%s", argv[2]);
	sprintf(pzOutput, "%s", argv[3]);
	sprintf(spacey, "%s", argv[4]);
	w = atoi(argv[5]);
	h = atoi(argv[6]);

	printf("pzInput=%s\r\n", pzInput);
	printf("space_in=%s\r\n", spacex);

	printf("pzOutput=%s\r\n", pzOutput);
	printf("space_out=%s\r\n", spacey);
	printf("w=%d\r\n", w);
	printf("h=%d\r\n", h);

	int S = 500;
	int T = 40;
	int F = 30;
	int n = w * h * F * T;

	void (*convert)(const float*, float*, int);

	if (strcmp(spacex, "normal") != 0) {
		printf("the input space must be normal!\r\n");
		return -1;
	}
	if (strcmp(spacey, "inverse")==0 ) {
		convert = & nrm2inv;
	} else if (strcmp(spacey, "log") == 0) {
		convert = & nrm2log;
	} else {
		printf("the output space must be either inverse or log!\r\n");
		return -1;
	}


	FILE* fin = fopen(pzInput, "rb");
	if (fin==0) {
		printf("Cannot open file: %s!\r\n", pzInput);
		return -1;
	}

   	FILE* fout = fopen(pzOutput, "wb");
	if (fout==0) {
		printf("Cannot open file: %s!\r\n", pzOutput);
		return -2;
	}

	size_t count=0;
	float* input = new float[n];
	float* output = new float[n];
	for (int s=0; s<S; s++){
		printf("processing scene %d....\r\n", s);

		// load data
		count = fread(input, n, sizeof(float), fin);
		if (count != n) {
			printf("reading data error: count=%d, n=%d......", count, n);
			return -1;
		}

		//convert 
		convert(input, output, n);

		// save data 
		count = fwrite(output, sizeof(float), n, fout);
		if (count != n) {
			printf("Writting data error: count=%d, n=%d......", count, n);
			return -1;
		}
	}
	delete[] input;
	delete[] output;

	fclose(fin);
	fclose(fout);
}

