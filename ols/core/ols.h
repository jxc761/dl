#ifndef NPLAB_OLS_H
#define NPLAB_OLS_H
#include <stdio.h>

namespace nplab{
	
	
	typedef float DepthType;
	
	typedef float FlowType;

	typedef unsigned char PixelType;
	
	// describe the object level scenes(v1)
	class OLS { 
	public:



#if defined(__APPLE__) || defined(__MACH__) // testing on local computer 

		
		static const int N_SCENES = 5;
		static const int N_TRACES = 2;
		static const int N_FRAMES = 10;
		
		static const int IMAGE_RES = 64;
		static const int DEPTH_RES = 256;
		static const int FLOW_RES  = 256;

#else  // run on real data

		static const int N_SCENES = 500;
		static const int N_TRACES = 40;
		static const int N_FRAMES = 30;
		
		static const int IMAGE_RES = 64;
		static const int DEPTH_RES = 256;
		static const int FLOW_RES  = 256;

#endif
		static const int MAX_PATH_LEN = 512;

		static const float NPLAB_INVALID_DEPTH;

		static const char PATH_TO_ROOT[];
		static const char PATH_TO_IMAGE[];
		static const char PATH_TO_DAE[];
		static const char PATH_TO_TRACE[];
		static const char TRACE_NAMES[40][16];

	public:


		static void getImageFileName(char* filename, int s, int t, int f) {

			sprintf(filename, "%s/scene_%d/%s/frame%03d.png", PATH_TO_IMAGE, s, TRACE_NAMES[t], f);
		}


		static void getTraceFileName(char* filename, int s, int t) {
			sprintf(filename, "%s/scene_%d/%s.ss.json", PATH_TO_TRACE, s, TRACE_NAMES[t]);
		}

		static void getDaeFileName(char* filename, int s) {
			sprintf(filename, "%s/scene_%d.dae", PATH_TO_DAE, s);
		}


	}; // end class OLS

}



#endif