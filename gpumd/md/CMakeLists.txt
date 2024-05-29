#####################################################################################
# MIT License                                                                       #
#                                                                                   #
# Copyright (c) 2024 Dr. Daniel Alejandro Matoz Fernandez                           #
#               fdamatoz@gmail.com                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#####################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)
project (ABPCPP C CXX)
set(CMAKE_BUILD_TYPE "Release")
# place all executables in the build directory 
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

########### ADD C++ SUPPORT  ##############
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)
########### SET C++ COMPILER FLAGS  ############
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ffast-math -DNDEBUG -std=c++14 -fvisibility=hidden -fpermissive" CACHE STRING "Flags used by the compiler during release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -DNDEBUG -std=c++14 -fvisibility=hidden -fpermissive" CACHE STRING "Flags used by the compiler during release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O -funroll-loops -ffast-math -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
############################################

########### ADD CUDA SUPPORT  ##############
find_package(CUDA 9.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
########### SET CUDA COMPILER FLAGS  ############
SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)
#list( APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list( APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg)
list( APPEND CUDA_NVCC_FLAGS --expt-extended-lambda )
list( APPEND CUDA_NVCC_FLAGS -std=c++14 )
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30 )
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 )
############################################


########### ADD PYBIND SUPPORT  ###############
add_subdirectory(external/pybind11)
set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "")
set(PYBIND11_CPP_STANDARD -std=c++14)
############################################


############ Add debugging option ##########
option(DEBUG_MODE "Enter debug mode" OFF)
if (DEBUG_MODE)
  add_definitions(-DDEBUG)
  MESSAGE("DEBUG MODE")
endif()
############################################

############ Add debugging option ##########
find_package(spdlog)

############################################

############################################
SET(INCLUDE_DIRS "${ABPCPP_SOURCE_DIR}/src")
include_directories(${INCLUDE_DIRS})
file(GLOB INCLUDE_DIRS_SRCS ${INCLUDE_DIRS}/*.cpp)
file(GLOB INCLUDE_DIRS_HDRS ${INCLUDE_DIRS}/*.hpp)
## utils ##
file(GLOB cuda_GPUHDRS ${INCLUDE_DIRS}/cuda/*.hpp)
file(GLOB cuda_GPUSRCS ${INCLUDE_DIRS}/cuda/*.cpp)
list(APPEND INCLUDE_DIRS_GPUHDRS ${cuda_GPUHDRS})
list(APPEND INCLUDE_DIRS_GPUSRCS ${cuda_GPUSRCS})
############################################

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
MESSAGE("\n") 
MESSAGE("sources : " ${INCLUDE_DIRS_SRCS}) 
MESSAGE("\n") 
MESSAGE("headers : " ${INCLUDE_DIRS_HDRS})
MESSAGE("\n") 
MESSAGE("gpu sources : " ${INCLUDE_DIRS_GPUSRCS}) 
MESSAGE("\n") 
MESSAGE("gpu headers : " ${INCLUDE_DIRS_GPUHDRS})
MESSAGE("\n") 

#this is needed to compile cpp files with nvcc
set_source_files_properties( ${INCLUDE_DIRS_GPUSRCS} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
#######################################################################

############ Add debugging option ##########
option(DEBUG_MODE "Enter debug mode" OFF)
if (DEBUG_MODE)
  add_definitions(-DDEBUG)
endif()
############################################

##########  LINK THE LIBRARIES CUDA-PYBIND  ##########
cuda_add_library(nvccmodule SHARED ${INCLUDE_DIRS_GPUSRCS} ${INCLUDE_DIRS_SRCS})
target_link_libraries(nvccmodule PRIVATE pybind11::module ${THREAD_LIB} ${CUDA_LIBRARIES} spdlog::spdlog cuda) 
set_target_properties(nvccmodule PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" OUTPUT_NAME "nvccmodule")                                               
#######################################################################
