import os
import sys
import numpy as np

# Determine the platform-specific bindings for the Sitnikov problem
if os.name == 'nt':
    sys.path.append(os.path.abspath("bindings/windows"))
    import PySitnikov
elif os.name == 'posix':
    sys.path.append(os.path.abspath("bindings/linux"))
    import PySitnikov
else:
    raise OSError("Unknown operating system")


class SitnikovProblem:
    def __init__(self):
        """Initializes the Sitnikov problem with default criteria values."""
        self.zCriteria = 10
        self.vzCriteria = 0.5
        self.azCriteria = 1e-2

    def run(self, e=0.0, z0=0.0, vz0=1.0, tFin=60.0, filename="", path="", returnTotal=False):
        """
        Runs the Sitnikov problem simulation with given parameters.

        :param e: Eccentricity
        :param z0: Initial z value
        :param vz0: Initial vz value
        :param tFin: Final time
        :param filename: Filename for the results
        :param path: Path for the results file
        :param returnTotal: Whether to return the complete results or not
        :return: Simulation results or the final iteration of the results, depending on returnTotal
        """
        # Initial conditions
        theta0 = 0
        y0 = [z0, vz0, theta0]
        t0 = 0
        h_initial = 0.01
        tolerance = 1e-8
        useCriterion = True

        # Construct the complete filename if a path is provided
        if path and not filename:
            filename = os.path.join(path, f"e={e}_z0={z0}_vz0={vz0}.txt")

        # Run the simulation
        sim = PySitnikov.runSitnikovSolver(t0, tFin, e, y0, h_initial, tolerance, self.zCriteria,
                                           self.vzCriteria, self.azCriteria, filename, useCriterion)

        # Return the complete results if requested, otherwise return the simulation data
        return readResults(filename) if returnTotal else sim


def readResults(filename):
    """
    Reads the simulation results from a given filename.

    :param filename: Name of the file to read the results from.
    :return: Arrays of time (t), z values, and vz values.
    """
    with open(filename, "r") as f:
        lines = f.readlines()[1:]
        t, z, vz = zip(*[(np.longdouble(val[0]), np.longdouble(val[2]), np.longdouble(val[3]))
                         for val in (line.split() for line in lines) if val[0].replace('.', '', 1).isdigit()])
    return np.array([t, z, vz])


if __name__ == '__main__':
    sitnikov = SitnikovProblem()
    sitnikov.run(e=0, z0=0, vz0=1, tFin=60, path="Simulations")
    print(readResults("Simulations/e=0_z0=0_vz0=1.txt").shape)
