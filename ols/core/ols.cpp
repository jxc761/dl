#include "ols.h"
namespace nplab{

#if defined(__APPLE__) || defined(__MACH__) // testing on local computer 
	const char OLS::PATH_TO_ROOT[]        = "/Users/Jing/Dropbox/dev/ols/data";
	const char OLS::PATH_TO_DAE[]         = "/Users/Jing/Dropbox/dev/ols/data/dae";
	const char OLS::PATH_TO_TRACE[]       = "/Users/Jing/Dropbox/dev/ols/data/trace/config_0";
	const char OLS::PATH_TO_IMAGE[]	      = "/Users/Jing/Dropbox/dev/ols/data/image/config_0";

#else // on hpc 
	const char OLS::PATH_TO_ROOT[]        = "/mnt/projects/CSE_CS_MSL88/object_level_scenes_v1";
	const char OLS::PATH_TO_DAE[]         = "/mnt/projects/CSE_CS_MSL88/object_level_scenes_v1/dae";
	const char OLS::PATH_TO_TRACE[]       = "/mnt/projects/CSE_CS_MSL88/object_level_scenes_v1/trace/config_0";
	const char OLS::PATH_TO_IMAGE[]	      = "/mnt/projects/CSE_CS_MSL88/object_level_scenes_v1/image/config_0";

#endif

	const char OLS::TRACE_NAMES[40][16] = { 
		"0_0_0_0", "0_0_1_0", "0_0_2_0", "0_0_3_0", "0_0_4_0", 
		"0_1_0_0", "0_1_1_0", "0_1_2_0", "0_1_3_0", "0_1_4_0", 
		"1_2_0_0", "1_2_1_0", "1_2_2_0", "1_2_3_0", "1_2_4_0", 
		"1_3_0_0", "1_3_1_0", "1_3_2_0", "1_3_3_0", "1_3_4_0", 
		"2_4_0_0", "2_4_1_0", "2_4_2_0", "2_4_3_0", "2_4_4_0", 
		"2_5_0_0", "2_5_1_0", "2_5_2_0", "2_5_3_0", "2_5_4_0", 
		"3_6_0_0", "3_6_1_0", "3_6_2_0", "3_6_3_0", "3_6_4_0", 
		"3_7_0_0", "3_7_1_0", "3_7_2_0", "3_7_3_0", "3_7_4_0"
	};
	const float OLS::NPLAB_INVALID_DEPTH=-1.0f;
}
