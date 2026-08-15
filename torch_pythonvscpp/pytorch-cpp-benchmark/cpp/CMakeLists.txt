cmake_minimum_required(VERSION 3.18)
project(libtorch_python_benchmark LANGUAGES CXX)

# CRITICAL: build against the SAME libtorch that ships with the Python wheel
# used on the Python side, or you are comparing builds, not languages.
# Configure with:
#   cmake -B build \
#     -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# On Isambard-AI build inside the same container / module environment that
# provides the sm_90 (Hopper) PyTorch build. Do NOT use a stray system
# libtorch.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Release optimisation for the C++ side. -DNDEBUG strips asserts.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(parity_check src/parity_check.cpp)
target_link_libraries(parity_check "${TORCH_LIBRARIES}")
set_property(TARGET parity_check PROPERTY CXX_STANDARD 17)

add_executable(harness src/harness.cpp)
target_link_libraries(harness "${TORCH_LIBRARIES}")
set_property(TARGET harness PROPERTY CXX_STANDARD 17)

# Report which libtorch we linked, so provenance is visible at configure time.
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries:    ${TORCH_LIBRARIES}")
