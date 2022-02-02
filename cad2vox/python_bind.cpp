#include <pybind11/pybind11.h>
// XTENSOR Python
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
#include "cpu_voxelizer.h"
namespace py = pybind11;
using namespace pybind11::literals;

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const Mesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool use_tetra, bool morton_code);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Encode 3d data using morton code, this is a mathematically efficient way of storing 3D data in binary.
// It is used for binary output in the c++ code but is not used for python.
// Thus is this is not used and is purely here for compatibility.
bool use_morton_code = false;

// Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed_tri(const Mesh *mesh){
  Timer t; t.start();
  size_t n_floats = sizeof(float) * 9 * (mesh->Surface.shape(0));
  float* device_triangles;
  int triangle_verts[3];
  fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
  checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
  fprintf(stdout, "[Mesh] Copy %zu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->Surface.shape(0)));
  for (size_t i = 0; i < mesh->Surface.shape(0); i++) {
    
	   // Extract the vertices that make up triangle i
    for (int K = 0; K < 3; K++){
	    triangle_verts[K] = mesh->Surface(i,K);
	  }
	  // get the xyz co-ordinates of each of those vertices as a glm vector
	  glm::vec3 v0 = glm::vec3(0,0,0);
	  glm::vec3 v1 = glm::vec3(0,0,0);
	  glm::vec3 v2 = glm::vec3(0,0,0);
	  
	  for (int N = 0; N < 3; N++){
	    v0[N] = mesh->Vertices(triangle_verts[0],N);
	    v1[N] = mesh->Vertices(triangle_verts[1],N);
	    v2[N] = mesh->Vertices(triangle_verts[2],N);
	  }
	  
	  size_t j = i * 9;
	  memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
	  memcpy((device_triangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
	  memcpy((device_triangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
	}
	t.stop();fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

	return device_triangles;
}

// Helper function to transfer tetrahedrons to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed_tets(const Mesh *mesh){
  Timer t; t.start();
  size_t n_floats = sizeof(float) * 12 * (mesh->Volume.shape(0));
  float* device_tets;
  int tetra_verts[4];
  fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for tetrahedron data \n", (readableSize(n_floats)).c_str());
  checkCudaErrors(cudaMallocManaged((void**) &device_tets, n_floats)); // managed memory
  fprintf(stdout, "[Mesh] Copy %zu tetrahedrons to CUDA-managed UNIFIED memory \n", (size_t)(mesh->Volume.shape(0)));
  for (size_t i = 0; i < mesh->Volume.shape(0); i++) {
    
	   // Extract the vertices that make up triangle i
    for (int K = 0; K < 4; K++){
	    tetra_verts[K] = mesh->Volume(i,K);
	  }
	  // get the xyz co-ordinates of each of those vertices as a glm vector
	  glm::vec3 A = glm::vec3(0,0,0);
	  glm::vec3 B = glm::vec3(0,0,0);
	  glm::vec3 C = glm::vec3(0,0,0);
	  glm::vec3 D = glm::vec3(0,0,0);
	  
	  for (int N = 0; N < 3; N++){
	    A[N] = mesh->Vertices(tetra_verts[0],N);
	    B[N] = mesh->Vertices(tetra_verts[1],N);
	    C[N] = mesh->Vertices(tetra_verts[2],N);
	    D[N] = mesh->Vertices(tetra_verts[3],N);
	  }
	  
	  size_t j = i * 12;
	  memcpy((device_tets)+j, glm::value_ptr(A), sizeof(glm::vec3));
	  memcpy((device_tets)+j+3, glm::value_ptr(B), sizeof(glm::vec3));
	  memcpy((device_tets)+j+6, glm::value_ptr(C), sizeof(glm::vec3));
	  memcpy((device_tets)+j+9, glm::value_ptr(D), sizeof(glm::vec3));
	}
	t.stop();fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

	return device_tets;
}

// Function to take in the result array and write greyscale values to it. This is done in parallel on the CPU at present.
xt::pyarray<unsigned char> write_greyscale(xt::pyarray<unsigned char> result,voxinfo info, unsigned int* vtable, Mesh* themesh){

  size_t voxels_seen = 0;
  const size_t write_stats_25 = (size_t(info.gridsize.x) * size_t(info.gridsize.y) * size_t(info.gridsize.z)) / 4.0f;
  fprintf(stdout, "writing greyscale values to array\n");
  
  for (size_t x = 0; x < info.gridsize.x; x++) {
    for (size_t y = 0; y < info.gridsize.y; y++) {
      for (size_t z = 0; z < info.gridsize.z; z++) {
	voxels_seen++;
	if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%...\n");}
	else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%...\n");}
	else if (voxels_seen == write_stats_25 * size_t(3)) {fprintf(stdout, "75%%...\n");}
	if (checkVoxel(x, y, z, info.gridsize, vtable)) {
	  result(x,y,z) = 255;
	}
      }
    }
  }
  return result;
}

xt::pyarray<unsigned char>run(xt::pyarray<long> Triangles, xt::pyarray<long> Tetra, xt::pyarray<long> Tags, xt::pyarray<float> Points,
		      xt::pyarray<float> Bbox_min, xt::pyarray<float> Bbox_max, unsigned int gridsize, bool useThrustPath = false,
		      bool forceCPU = false, bool solid = true, bool use_tetra=true){

  	Timer t; t.start();
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	fflush(stdout);
	xt::pyarray<unsigned char> result= xt::zeros<unsigned char>({gridsize,gridsize,gridsize});
	fprintf(stdout, "\n## READ MESH \n");
	
        Mesh *themesh = new Mesh(Triangles,Tetra,Tags,Points);
	int num_elem;
	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox

	glm::vec3 bbmin = Xt_to_glm(Bbox_min);
	glm::vec3 bbmax = Xt_to_glm(Bbox_max);
	glm::uvec3 voxgrid = glm::uvec3{gridsize,gridsize,gridsize};
	AABox<glm::vec3> bbox_mesh(bbmin,bbmax);
	if (use_tetra){
	  num_elem  = themesh->Volume.shape(0);
	}
	else {
	  num_elem = themesh->Surface.shape(0);
	  } 

	  
	  
	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), voxgrid, num_elem);
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
	unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this

	bool cuda_ok = false;
	if (!forceCPU)
	{
		// SECTION: Try to figure out if we have a CUDA-enabled GPU
		fprintf(stdout, "\n## CUDA INIT \n");
		cuda_ok = initCuda();
		cuda_ok ? fprintf(stdout, "[Info] CUDA GPU found\n") : fprintf(stdout, "[Info] CUDA GPU not found\n");
	}

	// SECTION: The actual voxelization
	if (cuda_ok && !forceCPU) { 
		// GPU voxelization
		fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

		float* device_elements;
		// Transfer triangles to GPU using either thrust or managed cuda memory
		if (useThrustPath && use_tetra) { device_elements = meshToGPU_thrust(themesh); }
		else { device_elements = meshToGPU_managed_tets(themesh); }
		
		if (useThrustPath && !use_tetra) { device_elements = meshToGPU_thrust(themesh); }
		else { device_elements = meshToGPU_managed_tri(themesh); }

		if (!useThrustPath) {
			fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
		}
		else {
			// ALLOCATE MEMORY ON HOST
			fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
		}
		fprintf(stdout, "\n## GPU VOXELISATION \n");
		if (solid){
			voxelize_solid(voxelization_info, device_elements, vtable, useThrustPath, use_morton_code);
		}
		else{
		  voxelize(voxelization_info, device_elements, vtable, useThrustPath, use_tetra, use_morton_code);
		}
		//result = write_greyscale(result,vtable);
	} else { 
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
		else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }

		if(use_tetra){
		  result = cpu_voxelizer::cpu_voxelize_volume(voxelization_info, themesh);
		}
		else if (!solid) {
		  result = cpu_voxelizer::cpu_voxelize_surface(voxelization_info, themesh, use_morton_code);
		  
		}
		else {
		  // allocate zero-filled array
		  vtable = (unsigned int*) calloc(1, vtable_size);
		  fprintf(stdout, "[WARN] Using option solid to auto-fill surface data.\n");
		  fprintf(stdout, "This option is quite slow and not very robust.\n Also custom greyscale values will be ignored.\n");
		  cpu_voxelizer::cpu_voxelize_surface_solid(voxelization_info, themesh, vtable, use_morton_code);
		  result = write_greyscale(result, voxelization_info, vtable, themesh);
		}
	}
      
	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
	return result;
}



PYBIND11_MODULE(_CudaVox, m) {
  
  xt::import_numpy();
    // Optional docstring
    m.doc() = "python  link into cudavox";
    m.def("run",&run,"function to perform the voxelization",
	  "Triangles"_a, "Tetra"_a, "Tags"_a,
	  "Points"_a, "Bbox_min"_a,
	  "Bbox_max"_a,"gridsize"_a, "useThrustPath"_a = false,
	  "forceCPU"_a = false, "solid"_a = false,
	  "use_tetra"_a =true);
}
