#pragma once

// GLM for maths
#include <glm/glm.hpp>
#include <cstdio>
#include <cmath> 
#include "util.h"
#include "timer.h"
#include "morton_LUTs.h"

class Mesh {
public:
	//
	// Types
	//
    xt::pyarray<long> Surface;
    xt::pyarray<long> Volume;
    xt::pyarray<long> Greyscale;
    xt::pyarray<float> Vertices;
// Constructor
    Mesh(xt::pyarray<long> triangle, xt::pyarray<long> tetra, xt::pyarray<long> greyscale, xt::pyarray<float> Points){
        Surface = triangle;
        Volume = tetra;
        Greyscale = greyscale;
        Vertices = Points;

}
};

namespace cpu_voxelizer {
	xt::pyarray<unsigned char> cpu_voxelize_surface(voxinfo info, Mesh* themesh, bool morton_order);
	void cpu_voxelize_surface_solid(voxinfo info, Mesh* themesh,unsigned int* vtable, bool morton_order);
    xt::pyarray<unsigned char> cpu_voxelize_volume(voxinfo info, Mesh* themesh);
}
