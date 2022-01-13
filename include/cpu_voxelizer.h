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
    xt::pyarray<long> Tags;
    xt::pyarray<float> Vertices;
// Constructor
    Mesh(xt::pyarray<long> triangle, xt::pyarray<long> tetra, xt::pyarray<long> tags, xt::pyarray<float> Points){
        Surface = triangle;
        Volume = tetra;
        Tags = tags;
        Vertices = Points;

}
};

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool** tri_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool morton_order);
}
