#ifndef NPLAB_GEOM_3D_H
#define NPLAB_GEOM_3D_H

#include <math.h>
namespace nplab{
	
	class Vector{ 
	public:
		double x, y, z; 
		Vector() {
			x = 0;
			y = 0;
			z = 0;
		}
		
		Vector(double _x, double _y, double _z) {
			x = (double) _x;
			y = (double) _y;
			z = (double) _z;
		}

		Vector(const Vector& v) {
			x = v.x;
			y = v.y;
			z = v.z;
		}

		Vector& operator=(const Vector& v) {
			x = v.x;
			y = v.y;
			z = v.z;

			return *this;
		}

		Vector& normalize() {
			double length = sqrt(x*x + y*y + z*z);
			x = x / length;
			y = y / length;
			z = z / length;
			return *this;
		}

	};


	typedef Vector Point;


	// /*------------------------------------------------------------------------*
	//  * Algebra part
	//  *------------------------------------------------------------------------*/
	// /**
	//  * w <- u x v
	//  */
	inline void cross_product(const Vector& u, const Vector& v, Vector* pw ){
		pw->x = u.y * v.z - u.z * v.y;
		pw->y = u.z * v.x - u.x * v.z;
		pw->z = u.x * v.y - u.y * v.x;
	}


	inline Vector operator*(double a, Vector& u) {
		return Vector(a * u.x, a * u.y, a * u.z);
	}

	inline Vector operator+(const Vector& u, const Vector& v) {
		return Vector(u.x+v.x, u.y+v.y, u.z+v.z);
	}

	inline Vector operator-(const Vector& u, const Vector& v) {
		return Vector(u.x - v.x, u.y-v.y, u.z-v.z);
	}

	inline double dot_product(const Vector& u, Vector& v) {
		return (u.x * v.x + u.y * v.y +  u.z * v.z  );
	}


	/*------------------------------------------------------------------------*
	* Convert from su
	*------------------------------------------------------------------------*/
	inline void ConvertFromSU(Point* pp){
		double x = pp->x, y=pp->y, z=pp->z;

		pp->x = x * 0.0254;
		pp->y = z * 0.0254;
		pp->z = -y * 0.0254;
	}


	
}

#endif