#include <pybind11/pybind11.h>
// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
#include "cpu_voxelizer.h"
namespace py = pybind11;

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const trimesh::TriMesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

//void run(string filename, bool useThrustPath =false, bool forceCPU = false, bool solid = false, unsigned int gridsize = 256);
// Encode 3d data using morton code, this is a mathematically efficient way of storing 3D data in binary.
// It is used for binary output in the c++ code but is not used for python.
// Thus is this is not used and is purely here for compatibility.
bool use_morton_code = false;

// Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh *mesh);

void check_filename(string filename){
  if (filename.empty()){
    throw "[ERROR] filename is empty!";
	    }
}

void run(string filename = "", bool useThrustPath = false, bool forceCPU = false, bool solid = false, unsigned int gridsize = 256){

  try{
    check_filename(filename);
      }
  catch (const char* msg) {
     cerr << msg << endl;
     exit(0);
   }
  	Timer t; t.start();
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	fflush(stdout);
	trimesh::TriMesh::set_verbose(false);

	// SECTION: Read the mesh from disk using the TriMesh library
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
	trimesh::TriMesh* themesh = trimesh::TriMesh::read(filename.c_str());
	
	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox
	AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));
	// Transform that AABox to a cubical box (by padding directions if needed)
	// Create voxinfo struct, which handles all the rest
	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), themesh->faces.size());
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
		if (!solid) {
		        cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, use_morton_code);
		}
		else {
		        cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, use_morton_code);
		}
	}

	//// DEBUG: print vtable
	for (int i = 0; i < vtable_size; i++) {
		char* vtable_p = (char*)vtable;
       	cout << (int) vtable_p[i] << endl;
	}

	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
}


PYBIND11_MODULE(CudaVox, m) {
    // Optional docstring
    m.doc() = "python  link into cudavox";
    m.def("test",&run,"function to perform the voxelization",
	  py::arg("filename") ="", py::arg("useThrustPath") = false,
	  py::arg("forceCPU") = false, py::arg("solid") = false,
	  py::arg("gridsize") = 256);
}
