#ifndef NPLAB_UTILS_H
#define NPLAB_UTILS_H

#include <stdarg.h> 
#include <stdio.h>
#include <stdlib.h>

#include "ols.h"

namespace nplab{

	inline void mkdir_for_file(const char* filename) {
		char cmd[1024];
		sprintf(cmd, "[[ -d $(dirname \"%s\") ]] || mkdir -p $(dirname \"%s\")", filename, filename);
		system(cmd);
	}
	
	inline void Error(const char* format, ...) {
		fprintf(stdout, "Error:\r\n");

		va_list argptr;
		va_start(argptr, format);
		vfprintf(stdout, format, argptr);
		va_end(argptr);


		fprintf(stdout, "\r\n");
		exit(EXIT_FAILURE);
	}
}

#endif