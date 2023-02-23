#include "voxelize.cuh"

// CUDA Global Memory variables

// Debug counters for some sanity checks
#ifdef _DEBUG
__device__ size_t debug_d_n_voxels_marked = 0;
__device__ size_t debug_d_n_triangles = 0;
__device__ size_t debug_d_n_voxels_tested = 0;
#endif

// Possible optimization: buffer bitsets (for now: Disabled because too much overhead)
//struct bufferedBitSetter{
//	unsigned int* voxel_table;
//	size_t current_int_location;
//	unsigned int current_mask;
//
//	__device__ __inline__ bufferedBitSetter(unsigned int* voxel_table, size_t index) :
//		voxel_table(voxel_table), current_mask(0) {
//		current_int_location = int(index / 32.0f);
//	}
//
//	__device__ __inline__ void setBit(size_t index){
//		size_t new_int_location = int(index / 32.0f);
//		if (current_int_location != new_int_location){
//			flush();
//			current_int_location = new_int_location;
//		}
//		unsigned int bit_pos = 31 - (unsigned int)(int(index) % 32);
//		current_mask = current_mask | (1 << bit_pos);
//	}
//
//	__device__ __inline__ void flush(){
//		if (current_mask != 0){
//			atomicOr(&(voxel_table[current_int_location]), current_mask);
//		}
//	}
//};

// Possible optimization: check bit before you set it - don't need to do atomic operation if it's already set to 1
// For now: overhead, so it seems
//__device__ __inline__ bool checkBit(unsigned int* voxel_table, size_t index){
//	size_t int_location = index / size_t(32);
//	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
//	return ((voxel_table[int_location]) & (1 << bit_pos));
//}

// readablesizestrings
//__device__ inline std::string readableSize(size_t bytes) {
//	double bytes_d = static_cast<double>(bytes);
//	std::string r;
//	if (bytes_d <= 0) r = "0 Bytes";
//	else if (bytes_d >= 1099511627776.0) r = std::to_string(static_cast<size_t>(bytes_d / 1099511627776.0)) + " TB";
//	else if (bytes_d >= 1073741824.0) r = std::to_string(static_cast<size_t>(bytes_d / 1073741824.0)) + " GB";
//	else if (bytes_d >= 1048576.0) r = std::to_string(static_cast<size_t>(bytes_d / 1048576.0)) + " MB";
//	else if (bytes_d >= 1024.0) r = std::to_string(static_cast<size_t>(bytes_d / 1024.0)) + " KB";
//	else r = std::to_string(static_cast<size_t>(bytes_d)) + " bytes";
//	return r;
// };

// Set a bit in the giant voxel table. This involves doing an atomic operation on a 32-bit word in memory.
// Blocking other threads writing to it for a very short time
__device__ __inline__ void setBit(unsigned int* voxel_table, size_t index){
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	atomicOr(&(voxel_table[int_location]), mask);
}

__device__ bool SameSideTri(glm::vec3 v1,glm::vec3 v2,glm::vec3 v3,glm::vec3 X,glm::vec3 p)
{
  // Edge vectors
  glm::vec3 e0 = v2 - v1;
  glm::vec3 e1 = v3 - v2;
  glm::vec3 e2 = v1 - v3;
  // Normal vector pointing up from the triangle
  glm::vec3 n = glm::normalize(glm::cross(e0, e1));
  //glm::vec3 normal = glm::cross(v2 - v1, v3 - v1);
  float dotX = glm::dot(n, X - v1);
  float dotP = glm::dot(n, p - v1);
  if (signbit(dotX)==signbit(dotP)||dotP==0){
    return true;
    }
  else
    {
    return false;
    }
}
  // check if the point P is inside the tetrahedron (v1,v2,v3,v4)
__device__ bool PointInTetrahedron(glm::vec3 v1, glm::vec3 v2,glm::vec3 v3,glm::vec3 v4,glm::vec3 p)
{
  return (SameSideTri(v1, v2, v3, v4, p) &&
	  SameSideTri(v2, v3, v4, v1, p) &&
	  SameSideTri(v3, v4, v1, v2, p) &&
	  SameSideTri(v4, v1, v2, v3, p));
}


// Main triangle voxelization method
__global__ void voxelize_triangle(voxinfo info, float* triangle_data, long* greyscale_data, unsigned int* voxel_table, unsigned short* result){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	// Common variables used in the voxelization process
	glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
	glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < info.n_triangles){ // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices
        size_t tri_index = size_t(t/9);
		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
		glm::vec3 v0 = glm::vec3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
		glm::vec3 v1 = glm::vec3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min; 
		glm::vec3 v2 = glm::vec3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;
		// Edge vectors
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<glm::ivec3> t_bbox_grid;
		t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);

		// PREPARE PLANE TEST PROPERTIES
		glm::vec3 c(0.0f, 0.0f, 0.0f);
		if (n.x > 0.0f) { c.x = info.unit.x; }
		if (n.y > 0.0f) { c.y = info.unit.y; }
		if (n.z > 0.0f) { c.z = info.unit.z; }
		float d1 = glm::dot(n, (c - v0));
		float d2 = glm::dot(n, ((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		glm::vec2 n_xy_e0(-1.0f*e0.y, e0.x);
		glm::vec2 n_xy_e1(-1.0f*e1.y, e1.x);
		glm::vec2 n_xy_e2(-1.0f*e2.y, e2.x);
		if (n.z < 0.0f) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * glm::dot(n_xy_e0, glm::vec2(v0.x, v0.y))) + glm::max(0.0f, info.unit.x*n_xy_e0[0]) + glm::max(0.0f, info.unit.y*n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * glm::dot(n_xy_e1, glm::vec2(v1.x, v1.y))) + glm::max(0.0f, info.unit.x*n_xy_e1[0]) + glm::max(0.0f, info.unit.y*n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * glm::dot(n_xy_e2, glm::vec2(v2.x, v2.y))) + glm::max(0.0f, info.unit.x*n_xy_e2[0]) + glm::max(0.0f, info.unit.y*n_xy_e2[1]);
		// YZ plane
		glm::vec2 n_yz_e0(-1.0f*e0.z, e0.y);
		glm::vec2 n_yz_e1(-1.0f*e1.z, e1.y);
		glm::vec2 n_yz_e2(-1.0f*e2.z, e2.y);
		if (n.x < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * glm::dot(n_yz_e0, glm::vec2(v0.y, v0.z))) + glm::max(0.0f, info.unit.y*n_yz_e0[0]) + glm::max(0.0f, info.unit.z*n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * glm::dot(n_yz_e1, glm::vec2(v1.y, v1.z))) + glm::max(0.0f, info.unit.y*n_yz_e1[0]) + glm::max(0.0f, info.unit.z*n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * glm::dot(n_yz_e2, glm::vec2(v2.y, v2.z))) + glm::max(0.0f, info.unit.y*n_yz_e2[0]) + glm::max(0.0f, info.unit.z*n_yz_e2[1]);
		// ZX plane
		glm::vec2 n_zx_e0(-1.0f*e0.x, e0.z);
		glm::vec2 n_zx_e1(-1.0f*e1.x, e1.z);
		glm::vec2 n_zx_e2(-1.0f*e2.x, e2.z);
		if (n.y < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x))) + glm::max(0.0f, info.unit.x*n_zx_e0[0]) + glm::max(0.0f, info.unit.z*n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x))) + glm::max(0.0f, info.unit.x*n_zx_e1[0]) + glm::max(0.0f, info.unit.z*n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x))) + glm::max(0.0f, info.unit.x*n_zx_e2[0]) + glm::max(0.0f, info.unit.z*n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++){
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
					// size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
					// if (checkBit(voxel_table, location)){ continue; }
#ifdef _DEBUG
					atomicAdd(&debug_d_n_voxels_tested, 1);
#endif
					// TRIANGLE PLANE THROUGH BOX TEST
					glm::vec3 p(x*info.unit.x, y*info.unit.y, z*info.unit.z);
					float nDOTp = glm::dot(n, p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f) { continue; }

					// PROJECTION TESTS
					// XY
					glm::vec2 p_xy(p.x, p.y);
					if ((glm::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f){ continue; }

					// YZ
					glm::vec2 p_yz(p.y, p.z);
					if ((glm::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f){ continue; }

					// XZ	
					glm::vec2 p_zx(p.z, p.x);
					if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f){ continue; }

#ifdef _DEBUG
					atomicAdd(&debug_d_n_voxels_marked, 1);
#endif

						size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y)* static_cast<size_t>(info.gridsize.y)) + (static_cast<size_t>(z)* static_cast<size_t>(info.gridsize.y)* static_cast<size_t>(info.gridsize.z));
						setBit(voxel_table, location);
                        result[location] = static_cast<unsigned short>(greyscale_data[tri_index]);
					continue;
				}
			}
		}
#ifdef _DEBUG
		atomicAdd(&debug_d_n_triangles, 1);
#endif
		thread_id += stride;
	}
}

// Main tetrahedron voxelization method
__global__ void voxelize_tetra(voxinfo info, float* tet_data, long* greyscale_data, unsigned int* voxel_table, unsigned short* result){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	// Common variables used in the voxelization process
	glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
	glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < info.n_triangles){ // every thread works on specific tetrahedron in its stride
		size_t t = thread_id * 12; // tetrahedron contains 12 vertices
        size_t tri_index = size_t(t/12);
		// COMPUTE COMMON TET PROPERTIES
		// Move vertices to origin using bbox
		glm::vec3 A = glm::vec3(tet_data[t], tet_data[t + 1], tet_data[t + 2]) - info.bbox.min;
		glm::vec3 B = glm::vec3(tet_data[t + 3], tet_data[t + 4], tet_data[t + 5]) - info.bbox.min; 
		glm::vec3 C = glm::vec3(tet_data[t + 6], tet_data[t + 7], tet_data[t + 8]) - info.bbox.min;
        glm::vec3 D = glm::vec3(tet_data[t + 9], tet_data[t + 10], tet_data[t + 11]) - info.bbox.min;
		

		// COMPUTE TETRA BBOX IN GRID
		// Tetrahedron bounding box in world coordinates is min(A,B,C,D) and max(A,B,C,D)
        AABox<glm::vec3> t_bbox_world(glm::min(A, glm::min(B, glm::min(C,D))),glm::max(A, glm::max(B, glm::max(C,D))));
		// Tetrahedron bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<glm::ivec3> t_bbox_grid;
		t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);

		
		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++){
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
					// size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
					// if (checkBit(voxel_table, location)){ continue; }
                    glm::vec3 P =  glm::vec3((x+0.5)*info.unit.x,(y+0.5)*info.unit.y,(z+0.5)*info.unit.z);
			        // check if point p is on the "correct" side of all 4 triangles and thus inside the tetrahedron.
			        if(PointInTetrahedron(A,B,C,D,P))
                    {
						size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y)* static_cast<size_t>(info.gridsize.y)) + (static_cast<size_t>(z)* static_cast<size_t>(info.gridsize.y)* static_cast<size_t>(info.gridsize.z));
						setBit(voxel_table, location);
                        result[location] = static_cast<unsigned short>(greyscale_data[tri_index]);
					}
					continue;
				    }
			    }
		    }
		thread_id += stride;
	    }
}

void voxelize(const voxinfo& v, float* element_data, long* greyscale_data, unsigned int* vtable, bool useThrustPath, unsigned short* result_array, bool use_tetra) {
	float elapsedTime;

	// These are only used when we're not using UNIFIED memory
	unsigned int* dev_vtable; // DEVICE pointer to voxel_data
    unsigned short* dev_result; // DEVICE pointer to voxel_data
	size_t vtable_size; // vtable size
	size_t result_size; // result size

	// Create timers, set start time
	cudaEvent_t start_vox, stop_vox;
	checkCudaErrors(cudaEventCreate(&start_vox));
	checkCudaErrors(cudaEventCreate(&stop_vox));

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxelize_triangle, 0, 0);
	// Round up according to array size 
	gridSize = (v.n_triangles + blockSize - 1) / blockSize;

	if (useThrustPath) { // We're not using UNIFIED memory
		vtable_size = ((size_t)v.gridsize.x * v.gridsize.y * v.gridsize.z) / (size_t) 8.0;
        result_size = ((size_t)v.gridsize.x * v.gridsize.y * v.gridsize.z)*sizeof(unsigned short);

		fprintf(stdout, "[Voxel Grid] Allocating %s of DEVICE memory for Voxel Grid\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaMalloc(&dev_vtable, vtable_size));
		checkCudaErrors(cudaMemset(dev_vtable, 0, vtable_size));

        fprintf(stdout, "[Voxel Grid] Allocating %s of DEVICE memory for Result\n", readableSize(result_size).c_str());
        checkCudaErrors(cudaMalloc(&dev_result, result_size));
		checkCudaErrors(cudaMemset(dev_result, 0, result_size));

		// Start voxelization
		checkCudaErrors(cudaEventRecord(start_vox, 0));
    if (use_tetra){voxelize_tetra << <gridSize, blockSize >> > (v, element_data, greyscale_data, dev_vtable, dev_result);}
	else {voxelize_triangle << <gridSize, blockSize >> > (v, element_data, greyscale_data, dev_vtable, dev_result);}
	}
	else { // UNIFIED MEMORY 
		checkCudaErrors(cudaEventRecord(start_vox, 0));
    if (use_tetra){voxelize_tetra << <gridSize, blockSize >> > (v, element_data, greyscale_data, vtable, result_array);}
	else {voxelize_triangle << <gridSize, blockSize >> > (v, element_data, greyscale_data, vtable, result_array);}
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaEventRecord(stop_vox, 0));
	checkCudaErrors(cudaEventSynchronize(stop_vox));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	printf("[Perf] Voxelization GPU time: %.1f ms\n", elapsedTime);

	// If we're not using UNIFIED memory, copy the voxel table back and free all
	if (useThrustPath){
		fprintf(stdout, "[Voxel Grid] Copying %s to page-locked HOST memory\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaMemcpy((void*)vtable, dev_vtable, vtable_size, cudaMemcpyDefault));
        fprintf(stdout, "[Voxel Grid] Copying %s to page-locked HOST memory\n", readableSize(result_size).c_str());
		checkCudaErrors(cudaMemcpy((void*)result_array, dev_result, result_size, cudaMemcpyDefault));
		fprintf(stdout, "[Voxel Grid] Freeing %s of DEVICE memory\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaFree(dev_vtable));
        fprintf(stdout, "[Voxel Grid] Freeing %s of DEVICE memory\n", readableSize(result_size).c_str());
		checkCudaErrors(cudaFree(dev_result));
	}

	// SANITY CHECKS
#ifdef _DEBUG
	size_t debug_n_triangles, debug_n_voxels_marked, debug_n_voxels_tested;
	checkCudaErrors(cudaMemcpyFromSymbol((void*)&(debug_n_triangles),debug_d_n_triangles, sizeof(debug_d_n_triangles), 0, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyFromSymbol((void*)&(debug_n_voxels_marked), debug_d_n_voxels_marked, sizeof(debug_d_n_voxels_marked), 0, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyFromSymbol((void*) & (debug_n_voxels_tested), debug_d_n_voxels_tested, sizeof(debug_d_n_voxels_tested), 0, cudaMemcpyDeviceToHost));
	printf("[Debug] Processed %zu triangles on the GPU \n", debug_n_triangles);
	printf("[Debug] Tested %zu voxels for overlap on GPU \n", debug_n_voxels_tested);
	printf("[Debug] Marked %zu voxels as filled (includes duplicates!) \n", debug_n_voxels_marked);
#endif

	// Destroy timers
	checkCudaErrors(cudaEventDestroy(start_vox));
	checkCudaErrors(cudaEventDestroy(stop_vox));
}
