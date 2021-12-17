CMAKE_MINIMUM_REQUIRED(VERSION 3.20 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
FIND_PACKAGE(glm CONFIG REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDAToolkit REQUIRED)
add_compile_options(-fPIC)
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")

IF(NOT Trimesh2_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_INCLUDE_DIR")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})

IF(NOT Trimesh2_TriMesh_h)
	message(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_TriMesh_h)

SET(Trimesh2_LINK_DIR CACHE PATH "Path to Trimesh2 library dir.")

IF(NOT Trimesh2_LINK_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_LINK_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
  MESSAGE(FATAL_ERROR "Trimesh2 library dir does not exist")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
	message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_LIBRARY)

MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")

SET(CUDA_VOXELIZER_SRCS
  ./src/util_cuda.cpp
  ./src/cpu_voxelizer.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/thrust_operations.cu
  ./src/voxelize_solid.cu
)

include_directories("include")



file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "src/*.hpp")
file (GLOB PYTHON_FILES "python/python_bind.cpp" "python/*.hpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(CudaVox
	${CUDA_VOXELIZER_SRCS}
    ${CUDA_VOXELIZER_SRCS_CU}
	${PYTHON_FILES}
)

TARGET_INCLUDE_DIRECTORIES(CudaVox PRIVATE ${Trimesh2_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(CudaVox PRIVATE ${Trimesh2_LIBRARY} PRIVATE OpenMP::OpenMP_CXX PRIVATE CUDA::cudart PRIVATE glm::glm)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#for tersting
set(PYTHON_LIBRARY_DIR "/home/ben/anaconda3/envs/CudaVox/lib")
set(PYTHON_EXECUTABLE "/home/ben/anaconda3/envs/CudaVox/bin/python3")

install(TARGETS CudaVox
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
