#pragma once

// GLM for maths
#include <glm/glm.hpp>
#include <cstdio>
#include <cmath> 
#include "util.h"
#include "timer.h"
#include "morton_LUTs.h"
// stuff for pybind11
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
// XTENSOR Python
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
//namespace py = pybind11;

class Mesh {
public:
	//
	// Types
	//
    xt::pyarray<float> Surface;
    xt::pyarray<float> Volume;
    xt::pyarray<float> Vertices;
// Constructor
    Mesh(xt::pyarray<float> triangle, xt::pyarray<float> tetra, xt::pyarray<float> Points){
        Surface = triangle;
        Volume = tetra;
        Vertices = Points;
}
};

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool** tri_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool morton_order);
}
