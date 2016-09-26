#ifndef NPLAB_CAMERA_H
#define NPLAB_CAMERA_H

#include "ols.h"
#include "geom.h"

namespace nplab{

	struct Camera {
		Vector  eye ; 	  ///< postion of camera
		Vector  target;  ///< postion of focus point
		Vector  up ;     ///< up_direction
		double  fov ;   ///< vetical fov of the camera
	};

	typedef struct Camera Camera;

	/* ------------------------------------------------------------------------
	 * Camera 
	 * Reference:
 	 * [1] P. Shirley and S. Marschner. 
 	 *     Fundamentals of Computer Graphics.
 	 *     3rd version 2011
	 *-------------------------------------------------------------------------
	 */

	/*----------------------------------------------------------------------
	 * W: reverse direction of gaze
	 * V: perpendicular to gaze direction and up direction
	 * U: at the same plane with up and gaze, but perpendicular to W and U.
	 * O; camera eye
	 *----------------------------------------------------------------------
	 */
	inline void compute_camera_coordinate(const Camera& camera, Vector* pU, Vector* pV, Vector* pW, Point* pO){
		Vector& U = *pU; 
		Vector& V = *pV;
		Vector& W = *pW;
		Point&  O = *pO;
		O = camera.eye;

		
		W = camera.eye - camera.target;
		W.normalize();

		cross_product(camera.up, W, &U);
		U.normalize();


		cross_product(W, U, &V);
		V.normalize();
	}

	/*----------------------------------------------------------------------
	 * Computing viewing rays from camera at each pixel in the image plane.
	 * camera[in]: the current camera state
	 * h[in] : image height
	 * w[in]: image width
	 * rays[out]: 
	 * stored in row-major, i.e. scanline from top to bottom
	 * the viewing ray at r-th row and c-th column is stored at rays[r*w+c]
	 *-----------------------------------------------------------------------
	 */
	inline void compute_viewing_rays(const Camera& camera, int h, int w, Vector* rays) {
		Vector U, V, W, O;
		compute_camera_coordinate(camera, &U, &V, &W, &O);
		double fov= camera.fov;
		const double c = tan(fov/2) / h;

		for (int x=0; x<w; x++) {
			double u = c * (2 * x + 1 - w);

			for (int y=0; y<h; y++) {
				double v = c * (h - 1 - 2 * y);
			
				Vector& ray = rays[ y * w + x ];
				ray = -1 * W + u * U + v * V;
				ray.normalize();
			}
		}
	}
}

#endif
