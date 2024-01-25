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

set(SOURCE_FILES_1 minmax.cpp)
add_executable(openmp_minmax ${SOURCE_FILES_1})

set(SOURCE_FILES_2 scalar.cpp)
add_executable(openmp_scalar ${SOURCE_FILES_2})

set(SOURCE_FILES_3 matrix.cpp)
add_executable(openmp_matrix ${SOURCE_FILES_3})

set(SOURCE_FILES_4 matrix2.cpp)
add_executable(openmp_matrix2 ${SOURCE_FILES_4})

set(SOURCE_FILES_5 scalar2.cpp)
add_executable(openmp_scalar2 ${SOURCE_FILES_5})
