#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
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

float* meshToGPU_managed_tets(const Mesh *mesh){
  Timer t; t.start();
  size_t n_floats = sizeof(float) * 12 * (mesh->Volume.shape(0));
  float* device_tets;
  int tet_verts[4];
  fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for tetrahedron data \n", (readableSize(n_floats)).c_str());
  checkCudaErrors(cudaMallocManaged((void**) &device_tets, n_floats)); // managed memory
  fprintf(stdout, "[Mesh] Copy %zu tetrahedrons to CUDA-managed UNIFIED memory \n", (size_t)(mesh->Volume.shape(0)));
  for (size_t i = 0; i < mesh->Volume.shape(0); i++) {
    
	   // Extract the vertices that make up tetrahedron i
    for (int K = 0; K < 4; K++){
	    tet_verts[K] = mesh->Volume(i,K);
	  }
	  // get the xyz co-ordinates of each of those vertices as a glm vector
	  glm::vec3 A = glm::vec3(0,0,0);
	  glm::vec3 B = glm::vec3(0,0,0);
	  glm::vec3 C = glm::vec3(0,0,0);
	  glm::vec3 D = glm::vec3(0,0,0);
	  
	  for (int N = 0; N < 3; N++){
	    A[N] = mesh->Vertices(tet_verts[0],N);
	    B[N] = mesh->Vertices(tet_verts[1],N);
	    C[N] = mesh->Vertices(tet_verts[2],N);
	    D[N] = mesh->Vertices(tet_verts[3],N);
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
// Helper function to transfer Greyscale values to automatically managed CUDA memory ( > CUDA 7.x)
long* GreyscaleToGPU_managed(const Mesh *mesh){
  Timer t; t.start();
  size_t n_floats = sizeof(long) * (mesh->Greyscale.shape(0));
  long* device_Greyscale;
  fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for Greyscasle data \n", (readableSize(n_floats)).c_str());
  checkCudaErrors(cudaMallocManaged((void**) &device_Greyscale, n_floats)); // managed memory
  fprintf(stdout, "[Mesh] Copy %zu Greyscale values to CUDA-managed UNIFIED memory \n", (size_t)(mesh->Greyscale.shape(0)));
  
  for (size_t i = 0; i < mesh->Greyscale.shape(0); i++) {
    *(device_Greyscale+i) = mesh->Greyscale(i);
	  }
  
	t.stop();fprintf(stdout, "[Perf] Greyscale transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

	return device_Greyscale;
}

//The "real" version of the function to try to figure out if we have a CUDA-enabled GPU. 
// There is a dummy version of this function (defined in cpu_voxeliser.h) that is for 
// when CUDA is not used during compilation. That function always return false to mimic
// this function failing to find a GPU.
bool Check_CUDA(){
  fprintf(stdout, "\n## CUDA INIT \n");
  bool cuda_ok = initCuda();
  cuda_ok ? fprintf(stdout, "[Info] CUDA GPU found\n") : fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n");
  return cuda_ok;
}

