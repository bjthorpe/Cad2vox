#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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

py::array_t<double>run(string filename = "", bool useThrustPath = false, bool forceCPU = false, bool solid = false, unsigned int gridsize = 256){

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
	trimesh::TriMesh *mesh = new trimesh::TriMesh();
   
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


	py::array_t<double> result = py::array_t<double>({gridsize,gridsize,gridsize});
        py::buffer_info info = result.request();
	auto ptr = static_cast<double *>(info.ptr); //pointer to the start of the array

	//N is the total numer of elemnts in the nparray r is a std::vector containing the number of elements in each dim
	int N = 1;
	for (auto r: info.shape) {
	  N *= r;
	}
	
	for (int n = 0; n < N; n++) {
	  int x = getx(n,gridsize,gridsize,gridsize);
	  int y = gety(n,gridsize,gridsize,gridsize);
	  int z = getz(n,gridsize,gridsize,gridsize);
	  
	  // the checkVoxel function returns either true or false which we can use as 0 or 1 in the
	  // array to save using if statements inside the loop.
	  *ptr = checkVoxel(x, y, z, voxelization_info.gridsize, vtable);
	  ptr++;
	    }
        
	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
	return result;
}


PYBIND11_MODULE(CudaVox, m) {
    // Optional docstring
    m.doc() = "python  link into cudavox";
    m.def("run",&run,"function to perform the voxelization",
	  py::arg("filename") ="", py::arg("useThrustPath") = false,
	  py::arg("forceCPU") = false, py::arg("solid") = false,
	  py::arg("gridsize") = 256);
}
