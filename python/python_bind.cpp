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
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Encode 3d data using morton code, this is a mathematically efficient way of storing 3D data in binary.
// It is used for binary output in the c++ code but is not used for python.
// Thus is this is not used and is purely here for compatibility.
bool use_morton_code = false;

// Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const Mesh *mesh){
  Timer t; t.start();
  size_t n_floats = sizeof(float) * 9 * (mesh->Surface.shape(0));
  float* device_triangles;
  int triangle_verts[3];
  fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
  checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
  fprintf(stdout, "[Mesh] Copy %lu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->Surface.shape(0)));
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

	//note: lx,ly and lz are the max length in each dim in this case = gridsize
	// unrool is the 
	int unroll(int x, int y, int z, int lx, int ly, int lz){
	  return x + y*lx + z*lx*ly;
	}

	int getx(int unrolled, int lx, int ly, int lz){
    // ly and lz not used - kept only for consistency
	  return unrolled % lx;
	}

	int gety(int unrolled, int lx, int ly, int lz){
    // lz not used - kept only for consistency
	  return (unrolled / lx) % ly;
	}

	int getz(int unrolled, int lx, int ly, int lz){
   // the last % lz should not be necessary
  // it is not used, by the way
	  return ((unrolled / lx) / ly) % lz;
	}


xt::pyarray<float>run(xt::pyarray<long> Triangles, xt::pyarray<long> Tetra, xt::pyarray<long> Tags, xt::pyarray<float> Points,
		      xt::pyarray<float> Bbox_min, xt::pyarray<float> Bbox_max, bool useThrustPath = false,
		      bool forceCPU = false, bool solid = true, unsigned int gridsize = 256, bool use_tetra=false){

  	Timer t; t.start();
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	fflush(stdout);
	xt::pyarray<long> result;
	fprintf(stdout, "\n## READ MESH \n");
	
        Mesh *themesh = new Mesh(Triangles,Tetra,Tags,Points);

	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox

	glm::vec3 bbmin = Xt_to_glm(Bbox_min);
	glm::vec3 bbmax = Xt_to_glm(Bbox_max);
	AABox<glm::vec3> bbox_mesh(bbmin,bbmax);
	// Transform that AABox to a cubical box (by padding directions if needed)
	// Create voxinfo struct, which handles all the rest
	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), themesh->Surface.shape(0));
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
	unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this
	bool (**tri_table);

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

		float* device_triangles;
		// Transfer triangles to GPU using either thrust or managed cuda memory
		if (useThrustPath) { device_triangles = meshToGPU_thrust(themesh); }
		else { device_triangles = meshToGPU_managed(themesh); }

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
			voxelize_solid(voxelization_info, device_triangles, vtable, useThrustPath, use_morton_code);
		}
		else{
			voxelize(voxelization_info, device_triangles, vtable, useThrustPath, use_morton_code);
		}
	} else { 
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
		else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
		// allocate zero-filled array
		vtable = (unsigned int*) calloc(1, vtable_size);
		tri_table = (bool**)calloc(voxelization_info.n_triangles,sizeof(bool*));
		for(int i = 0; i < 100; ++i) {

		  tri_table[i] = (bool *) calloc(vtable_size, sizeof(bool));
		}
		
		if (!solid) {
		  cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, tri_table, use_morton_code);
		}
		else {
		        cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, use_morton_code);
		}
	}

	// Generate output without greyscale
	  result= xt::zeros<long>({gridsize,gridsize,gridsize});

	  for (int x = 0; x < gridsize; x++) {
	    for (int y = 0; y < gridsize; y++) {
	      for (int z = 0; z < gridsize; z++) {
		if(checkVoxel(x, y, z, voxelization_info.gridsize, vtable)){
		  result(x,y,z) = 256; // set voxel to max brightness (8bit rgb)
		}
	      }
	    }
	  }
	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
	return result;
}



PYBIND11_MODULE(CudaVox, m) {
  
  xt::import_numpy();
    // Optional docstring
    m.doc() = "python  link into cudavox";
    m.def("run",&run,"function to perform the voxelization",
	  "Triangles"_a, "Tetra"_a, "Tags"_a,
	  "Points"_a, "Bbox_min"_a,
	  "Bbox_max"_a, "useThrustPath"_a = false, "forceCPU"_a = false,
	  "solid"_a = true,"gridsize"_a = 256, "use_tetra"_a =false);
}
