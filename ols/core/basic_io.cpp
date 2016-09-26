

#include <stdlib.h>
#include <stdio.h>
#include <fstream>

#include "json.hpp"
#include "ols.h"
#include "geom.h"
#include "camera.h"
#include "utils.h"


namespace nplab{

	void ols_load_cameras(int s, int t, Camera* cameras) {
		float fov = 2* atan(0.012/0.050);
		int   n   = OLS::N_FRAMES;

		char filename[512];
		OLS::getTraceFileName(filename, s, t);

		nlohmann::json ss;

		// read in 
		std::ifstream instream (filename, std::ifstream::in);
		if (!instream.bad()) {
			instream >> ss;
			instream.close();
		} else {
			Error("location: ols_load_cameras()\r\n Cann't load file in:%s!", filename);
		}
		
		// target 
		nlohmann::json::array_t pos = ss["target"]["position"]["origin"];
		Vector target(pos[0],pos[1],pos[2]);
		ConvertFromSU(&target);

		//trace
	 	nlohmann::json::array_t trace =  ss["camera_trajectory"]["trace"].get<nlohmann::json::array_t>();

	    for (int i = 0; i < n; i++)
	    {
	    	// printf("camera: %d\r\n", i);
	    	nlohmann::json::object_t cur = trace[i];
	    	Camera& cam = cameras[i];

	    	cam.target = target;


	    	nlohmann::json::array_t eye = cur["origin"];

	    	cam.eye.x = eye[0];
	    	cam.eye.y = eye[1];
	    	cam.eye.z = eye[2];
	    	ConvertFromSU(& (cam.eye) );


			
			nlohmann::json::array_t up = cur["zaxis"];
	    	cam.up.x  = up[0];
	    	cam.up.y  = up[1];
	    	cam.up.z  = up[2];
	    	ConvertFromSU(& (cam.up) );

	    	cam.fov = fov;
	    	//printf("cam(%d)=(%.2f, %.2f, %.2f)\r\n", i, cam.eye.x, cam.eye.y, cam.eye.z);
	    }

	}


}
