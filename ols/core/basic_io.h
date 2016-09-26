#ifndef NPLAB_BASIC_IO_H
#define NPLAB_BASIC_IO_H

#include "camera.h"

namespace nplab{

	void mkdir_for_file(const char* filename);

	void ols_load_cameras(int s, int t, Camera* cameras);
}


#endif