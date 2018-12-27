cmake_minimum_required(VERSION 3.13)
project(LFP_Decoding)

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(include)
LINK_DIRECTORIES(lib)

add_executable(LFP_Decoding main.cpp algorithm.cpp)

target_link_libraries(LFP_Decoding libmtpsd.a libdpss.a)
target_link_libraries(LFP_Decoding libfftw3.a)
target_link_libraries(LFP_Decoding liblapack.a librefblas.a libtmglib.a)
target_link_libraries(LFP_Decoding gfortran pthread)
