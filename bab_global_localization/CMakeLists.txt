cmake_minimum_required(VERSION 2.8.3)
project(bab_global_localization_github)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

pkg_check_modules(YAMLCPP yaml-cpp REQUIRED)
if(YAMLCPP_VERSION VERSION_GREATER "0.5.0")
    add_definitions(-DHAVE_YAMLCPP_GT_0_5_0)
endif()

# include directories
include_directories(
	include
	${PCL_INCLUDE_DIRS}
	/usr/include/eigen3
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(bab_global_localization src/bab_global_localization.cpp)
target_compile_options(bab_global_localization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(bab_global_localization ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenCV_LIBS} yaml-cpp)

