#include "thrust_operations.cuh"
#include "cpu_voxelizer.h"
// thrust vectors (global) (see https://stackoverflow.com/questions/54742267/having-thrustdevice-vector-in-global-scope)
thrust::host_vector<glm::vec3>* trianglethrust_host;
thrust::device_vector<glm::vec3>* trianglethrust_device;

// method 3: use a thrust vector
float* meshToGPU_thrust(const Mesh *mesh) {
	Timer t; t.start(); // TIMER START
	// create vectors on heap 
	trianglethrust_host = new thrust::host_vector<glm::vec3>;
	trianglethrust_device = new thrust::device_vector<glm::vec3>;
    int triangle_verts[3];
	// fill host vector
	fprintf(stdout, "[Mesh] Copying %zu triangles to Thrust host vector \n", mesh->Surface.shape(0));
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
		
        //glm::vec3 v0 = Xt_to_glm<float>(mesh->Vertices(mesh->Surface(i,0)));
		//glm::vec3 v1 = Xt_to_glm<float>(mesh->Vertices(mesh->Surface(i,1)));
		//glm::vec3 v2 = Xt_to_glm<float>(mesh->Vertices(mesh->Surface(i,2)));
		trianglethrust_host->push_back(v0);
		trianglethrust_host->push_back(v1);
		trianglethrust_host->push_back(v2);
	}
	fprintf(stdout, "[Mesh] Copying Thrust host vector to Thrust device vector \n");
	*trianglethrust_device = *trianglethrust_host;
	t.stop(); fprintf(stdout, "[Mesh] Transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds); // TIMER END
	return (float*) thrust::raw_pointer_cast(&((*trianglethrust_device)[0]));
}

void cleanup_thrust(){
	fprintf(stdout, "[Mesh] Freeing Thrust host and device vectors \n");
	if (trianglethrust_device) free(trianglethrust_device);
	if (trianglethrust_host) free(trianglethrust_host);
}
