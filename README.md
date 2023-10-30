# SitnikovSolver

The SitnikovSolver is a high-performance solver for the Sitnikov problem, utilizing the Runge-Kutta-Fehlberg 7(8) integration scheme. It's implemented in C++ for optimal speed, and features a Python interface for versatility and ease of use.

## Prerequisites

- Python 3.10 (Note: Bindings are provided for both Linux and Windows. However, due to potential compatibility issues between different Python versions, it is recommended to compile the code specifically for your environment.)
- Pybind11 package (used for creating Python bindings to the C++ code)

## Building

1. Navigate to the `SitnikovCpp` directory.
2. Update the path to the Pybind11 package in `CMakeLists.txt` to reflect your environment.
3. Follow the build commands provided in `SitnikovCpp/Commands.txt` to compile the solver.

> **Tip**: If you are familiar with other binding tools like `cppyy`, you can use them to create your own bindings based on your preferences and needs.
