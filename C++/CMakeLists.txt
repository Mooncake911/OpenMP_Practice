cmake_minimum_required(VERSION 3.27)
project(Projects_MP)

enable_language(CXX)

if (MSVC)
    set(CMAKE_CXX_COMPILER cl)
    set(CMAKE_CXX_STANDARD 17)

    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -openmp:llvm -openmp:experimental")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCE_FILES your.cpp)
add_executable(openmp_test_clion ${SOURCE_FILES})
