#ifndef NPLAB_DEPTH_DS_UTILS_H
#define NPLAB_DEPTH_DS_UTILS_H


void enter_embree();
void exit_embree();

// for cache out depth map
void cache_scene(int s, int w, int h, FILE* pout);


#endif


