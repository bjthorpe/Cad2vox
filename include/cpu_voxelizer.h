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
#include <pybind11/eigen.h>
//using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
typedef Eigen::Matrix<float,3, Eigen::Dynamic> MatrixX3f;
// Use RowMatrixXf instead of MatrixXf to map numpy arrays to eigen
// Eigen::Ref<RowMatrixXf>
namespace py = pybind11;

class Mesh {
public:
	//
	// Types
	//
    MatrixX3f Surface;
    MatrixX3f Volume;
    MatrixX3f Vertices;
// Constructor
    Mesh(Eigen::MatrixX3f triangle, Eigen::MatrixX3f tetra, Eigen::MatrixX3f Points){
        Surface = triangle;
        Volume = tetra;
        Vertices = Points;
}
};

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool** tri_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, Mesh* themesh, unsigned int* voxel_table, bool morton_order);
}
