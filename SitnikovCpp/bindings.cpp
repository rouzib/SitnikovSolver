#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sitnikov.cpp"

namespace py = pybind11;

PYBIND11_MODULE(PySitnikov, m
) {
    m.doc() = "pybind11 Sitnikov problem plugin"; // optional module docstring
    m.def("runSitnikovSolver", &runSitnikovSolver,
          R"pbdoc(
          Function to set up the Sitnikov problem and run the RKF 7(8) solver.

          Parameters:
          - t0: Initial time (default: 0.0).
          - t_final: Final time (default: 60.0).
          - e: Eccentricity parameter (default: 0.0).
          - y0: Initial state vector [z, vz, theta] (default: {0.0, 1.0, 0.0}).
          - h_initial: Initial step size for the RKF 7(8) solver (default: 0.01).
          - tolerance: Error tolerance for the RKF 7(8) solver (default: 1e-8).
          - zCriteria: Stopping criteria based on z value (default: 10).
          - vzCriteria: Stopping criteria based on vz value (default: 0.5).
          - azCriteria: Stopping criteria based on az value (default: 0.01).
          - filename: The name of the file to output to (default: "output.txt").
          - useCriterion: If the criterion should be used (default: true).

          Returns:
          - Final state vector after integrating up to t_final.
          )pbdoc",
          py::arg("t0") = 0.0,
          py::arg("t_final") = 60.0,
          py::arg("e") = 0.0,
          py::arg("y0") = std::vector<double>{
                  0.0, 1.0, 0.0},
          py::arg("h_initial") = 0.01,
          py::arg("tolerance") = 1e-8,
          py::arg("zCriteria") = 10,
          py::arg("vzCriteria") = 0.5,
          py::arg("azCriteria") = 0.01,
          py::arg("filename") = "output.txt",
          py::arg("useCriterion") = true
    );
}