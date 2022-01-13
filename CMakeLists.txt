CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
FIND_PACKAGE(glm CONFIG REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDAToolkit REQUIRED)
FIND_PACKAGE(xtensor REQUIRED)

add_compile_options(-g)
#set(XTL_INCLUDE_DIRS ${CONDA_PREFIX}/include lib/xtensor-python/include)
set(XTL_INCLUDE_DIRS lib/xtensor-python/include)
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(CUDA_VOXELIZER_SRCS
  ./src/util_cuda.cpp
  ./src/cpu_voxelizer.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/thrust_operations.cu
  ./src/voxelize_solid.cu
)

#include_directories("include")

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "src/*.hpp")
file (GLOB PYTHON_FILES "python/python_bind.cpp" "python/*.hpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(CudaVox
	${CUDA_VOXELIZER_SRCS}
    ${CUDA_VOXELIZER_SRCS_CU}
	${PYTHON_FILES}
)

TARGET_LINK_LIBRARIES(CudaVox PRIVATE OpenMP::OpenMP_CXX PRIVATE CUDA::cudart PRIVATE glm::glm PRIVATE xtensor)
target_include_directories(CudaVox PUBLIC ${XTL_INCLUDE_DIRS} PUBLIC "include")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#for testing
set(PYTHON_LIBRARY_DIR "/home/ben/anaconda3/envs/CudaVox/lib")
set(PYTHON_EXECUTABLE "/home/ben/anaconda3/envs/CudaVox/bin/python3")

install(TARGETS CudaVox
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
