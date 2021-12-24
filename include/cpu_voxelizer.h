#pragma once

// GLM for maths
#include <glm/glm.hpp>
#include <cstdio>
#include <cmath> 
#include "util.h"
#include "timer.h"
#include "morton_LUTs.h"
// stuff for pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class Mesh {
public:
	//
	// Types
	//
    py::array_t<float> Surface;
    py::array_t<float> Volume;
    py::array_t<float> Vertices;
// Constructor
    Mesh(py::array_t<float> triangle, py::array_t<float> tetra, py::array_t<float> Points){
        Surface = triangle;
        Volume = tetra;
        Vertices = Points;
}
};

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool** tri_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool morton_order);
}
