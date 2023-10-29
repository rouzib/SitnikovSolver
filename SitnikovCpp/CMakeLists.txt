cmake_minimum_required(VERSION 3.21)
project(SitnikovSolver)


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(pybind11Path "C:/Users/npayo/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(pybind11Path "/home/rouzib/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_PREFIX_PATH ${pybind11Path} ${CMAKE_PREFIX_PATH})
find_package(pybind11 REQUIRED)

option(BUILD_PYTHON_MODULE "Build the Python module" ON)

if(BUILD_PYTHON_MODULE)
    # Commands to build the Python module
    message(STATUS "Building the Python module...")
    pybind11_add_module(PySitnikov bindings.cpp)
    target_link_libraries(PySitnikov PRIVATE pybind11::embed)
else()
    # Commands to build the standalone executable
    message(STATUS "Building the standalone executable...")
    add_executable(SitnikovExe sitnikov.cpp)
endif()