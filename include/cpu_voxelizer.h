#pragma once

// GLM for maths
#include <glm/glm.hpp>
#include <cstdio>
#include <cmath> 
#include "util.h"
#include "timer.h"

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

#ifndef WITH_CUDA
// Dummy Function to be called when cuda is not being used to simply print out a message and return false.
// The equivlent function to call when using CUDA is defined in managed_mem.h"
bool Check_CUDA(){
    fprintf(stdout, "\n[Info] CUDA was not included when compiled.  \n");
    fprintf(stdout, "[Info] Running on CPU with OpenMP.\n");
    return false;
}
#endif

namespace cpu_voxelizer {
	xt::pyarray<unsigned char> cpu_voxelize_surface(voxinfo info, Mesh* themesh);
	void cpu_voxelize_surface_solid(voxinfo info, Mesh* themesh,unsigned int* vtable);
    xt::pyarray<unsigned char> cpu_voxelize_volume(voxinfo info, Mesh* themesh);
}
