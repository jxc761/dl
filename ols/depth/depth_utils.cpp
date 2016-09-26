
#include <float.h> // FLT_MAX


// for cache-data
#include <stdio.h>
#include "utils.h"


#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "ols.h"
#include "geom.h"
#include "camera.h"
#include "basic_io.h"


#include "depth_utils.h"

using namespace nplab;

/*----------------------------------------------------------------------------
 *
 *---------------------------------------------------------------------------*/

static RTCDevice g_rtc_device=NULL;
void embree_error_handler(const RTCError code, const char* str) ;

/* error reporting function */
void embree_error_handler(const RTCError code, const char* str) {
	printf("Embree: ");
	switch (code) {
		case RTC_UNKNOWN_ERROR    : printf("RTC_UNKNOWN_ERROR"); break;
		case RTC_INVALID_ARGUMENT : printf("RTC_INVALID_ARGUMENT"); break;
		case RTC_INVALID_OPERATION: printf("RTC_INVALID_OPERATION"); break;
		case RTC_OUT_OF_MEMORY    : printf("RTC_OUT_OF_MEMORY"); break;
		case RTC_UNSUPPORTED_CPU  : printf("RTC_UNSUPPORTED_CPU"); break;
		case RTC_CANCELLED        : printf("RTC_CANCELLED"); break;
		default                   : printf("invalid error code"); break;
	}
	if (str) { 
		printf(" ("); 
		while (*str) putchar(*str++); 
		printf(")\n"); 
	}
	exit(1);
}


void enter_embree() {
	/* initialize ray tracing core */
	// rtcInit(NULL);
	// RTCDevice device = rtcNewDevice(NULL);
	if (g_rtc_device == NULL) {
		 g_rtc_device = rtcNewDevice(NULL);
	}

	/* initialize ray tracing core */
	/* for best performance set FTZ and DAZ flags in MXCSR control and status register */
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);



	/* set error handler */
	// rtcSetErrorFunction(embree_error_handler);
	rtcDeviceSetErrorFunction(g_rtc_device, embree_error_handler);
}


void exit_embree() {
	if (g_rtc_device != NULL) {
 		rtcDeleteDevice(g_rtc_device);
 	}
}


/*----------------------------------------------------------------------------
 *
 *---------------------------------------------------------------------------*/
struct RTCVertex   { float x, y, z, a; };
struct RTCTriangle { int v0, v1, v2; };

// load the scene from file
static int load_3dmodel(RTCScene scene, const char* pzInput){
	// Create an instance of the Importer class
	Assimp::Importer importer;
	
	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll 
	// propably to request more postprocessing than we do in this example.
	const aiScene* org = importer.ReadFile( pzInput, 
		aiProcess_CalcTangentSpace       | 
		aiProcess_Triangulate            |
		aiProcess_JoinIdenticalVertices  |
		aiProcess_PreTransformVertices   | 
		aiProcess_SortByPType);

	// If the import failed, report it
	if( !org)
	{
		// DoTheErrorLogging( importer.GetErrorString());
		printf( "Can not read : %s \r\n", pzInput);
		return 0;
	} 


	// Now we can access the file's contents. 
	/* process mesh one by one */
	for ( unsigned int i = 0; i <  org ->mNumMeshes; i++ ) {

		aiMesh * pOrgMesh = org->mMeshes[i];

        if ( pOrgMesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE ) {
            continue;
        }

		unsigned int nVertices =  pOrgMesh->mNumVertices; 
		unsigned int nFaces  = pOrgMesh->mNumFaces; 
		
		/* create a triangulated plane with nFaces triangles and nVertices vertices */
  		unsigned int dstMesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, nFaces, nVertices);

  		/* set vertices */
  		RTCVertex* vertices = (RTCVertex*) rtcMapBuffer(scene, dstMesh, RTC_VERTEX_BUFFER); 
		for (unsigned int j = 0; j < nVertices; j++) {

			vertices[j].x = pOrgMesh->mVertices[j].x; 
			vertices[j].y = pOrgMesh->mVertices[j].y; 
			vertices[j].z = pOrgMesh->mVertices[j].z; 
			// printf("v%d: (%.4f, %.4f, %.4f) \r\n",  j, vertices[j].x,  vertices[j].y,  vertices[j].z); 
		}
  		rtcUnmapBuffer(scene, dstMesh, RTC_VERTEX_BUFFER); 

		/* set triangles */
		RTCTriangle* triangles = (RTCTriangle*) rtcMapBuffer(scene, dstMesh, RTC_INDEX_BUFFER);
		for (unsigned int j = 0; j < nFaces; j++) {
	
			triangles[j].v0 = pOrgMesh->mFaces[j].mIndices[0];
			triangles[j].v1 = pOrgMesh->mFaces[j].mIndices[1];
			triangles[j].v2 = pOrgMesh->mFaces[j].mIndices[2];	
		}
		rtcUnmapBuffer(scene, dstMesh, RTC_INDEX_BUFFER);

		// printf("mesh %d: %d faces, %d vertices \r\n", dstMesh, nFaces, nVertices);
	}


	// We're done. Everything will be cleaned up by the importer destructor
	return 1;

}

static void cal_depth(RTCScene scene, Camera& camera, int height, int width, DepthType* depth){

	// origin of rays
	Point origin;
	origin.x = camera.eye.x;
	origin.y = camera.eye.y;
	origin.z = camera.eye.z;
	
	// directions of rays
	Vector* directions = new Vector[height * width];
	// cal_ray_directions(camera, height, width, directions);
	compute_viewing_rays(camera, height, width, directions);

	// compute rays intersection
	int i = 0, j=0;  
	// float hi = 0, wj = 0;
	for (i = 0; i < height ; i++ ){
		// hi = top - delta * i; 

		for (j = 0; j < width; j++){
			// wj = left - delta * j;
			
			RTCRay ray;

			ray.org[0] = origin.x;
			ray.org[1] = origin.y;
			ray.org[2] = origin.z; 

			ray.dir[0] = directions[i * width + j].x;
			ray.dir[1] = directions[i * width + j].y;
			ray.dir[2] = directions[i * width + j].z;
			
			ray.tnear = 0.f;
			ray.tfar =  FLT_MAX;		// in & out
			ray.geomID = RTC_INVALID_GEOMETRY_ID; // out params of rtcIntersect 
			ray.primID = RTC_INVALID_GEOMETRY_ID; //// out params of rtcIntersect 
			ray.instID = RTC_INVALID_GEOMETRY_ID;
			ray.mask = 0xFFFFFFFF; // ray active 
			ray.time = 0.f;

			

			rtcIntersect(scene, ray); 
	

			depth[i * width + j] = (ray.geomID != RTC_INVALID_GEOMETRY_ID) ? ray.tfar : OLS::NPLAB_INVALID_DEPTH;
		}
	}

	delete[] directions;

}

// 
void cache_scene(int s, int w, int h, FILE* pout){
	int T = OLS::N_TRACES;
	int F = OLS::N_FRAMES;
	int n = w*h;
	/* 
	 * load scene in 
	 */	
	char filename[512];
	OLS::getDaeFileName(filename, s);

	/* create scene */
	//RTCScene scene  = rtcNewScene(RTC_SCENE_STATIC, RTC_INTERSECT1);
	if(g_rtc_device==NULL) {
		printf("device have been released\r\n");
	}

	RTCScene scene  = rtcDeviceNewScene(g_rtc_device, RTC_SCENE_STATIC, RTC_INTERSECT1);
	
	/* build scene */
	load_3dmodel(scene, filename);
		
	/* commit changes to scene */
	rtcCommit (scene);

	Camera* cameras= new Camera[F];
	DepthType* depthmap = new DepthType[n];

	for (int t=0; t < T; t++) {
		// load the cameras on each trace in
		ols_load_cameras(s, t, cameras);

		for ( int f=0; f < F; f++) {
			// compute the depth map for each frame
			cal_depth(scene, cameras[f], h, w, depthmap);
			
			// save out result
			size_t count= fwrite( depthmap, sizeof(DepthType), n, pout);
		    if(count != n) {
			   Error("fwrite error in  cache_scene(s=%d, t=%d, f=%d, w=%d, h=%d): %d != %d. \r\n ", s, t, f, w, h, count, n);
			}
		}
	}
	
	rtcDeleteScene (scene);
}



