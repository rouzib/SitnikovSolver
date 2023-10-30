import os
import sys
import logging
import numpy as np

# Determine the platform-specific bindings for the Sitnikov problem
if os.name == 'nt':
    sys.path.append(os.path.abspath("../../SitnikovCpp/bindings/windows"))
    import PySitnikov
elif os.name == 'posix':
    sys.path.append(os.path.abspath("../../SitnikovCpp/bindings/linux"))
    import PySitnikov
else:
    raise OSError("Unknown operating system")


class SitnikovProblem:
    def __init__(self):
        """Initializes the Sitnikov problem with default criteria values."""
        self.zCriteria = 10
        self.vzCriteria = 0.5
        self.azCriteria = 1e-2

    def run(self, e=0.0, z0=0.0, vz0=1.0, tFin=60.0, filename="", path="", returnTotal=False, debug=False):
        """
        Runs the Sitnikov problem simulation with given parameters.

        :param e: Eccentricity
        :param z0: Initial z value
        :param vz0: Initial vz value
        :param tFin: Final time
        :param filename: Filename for the results
        :param path: Path for the results file
        :param returnTotal: Whether to return the complete results or not
        :param debug: Show message indicating it is simulating
        :return: Simulation results or the final iteration of the results, depending on returnTotal
        """
        if debug:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Simulating e={e}, z0={z0}, vz0={vz0}")
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

    def runForAL(self, e=0.0, z0=0.0, vz0=1.0, tFin=60.0, path=""):
        """
        Runs the Sitnikov problem simulation with given parameters. Intended for the AL code

        :param e: Eccentricity
        :param z0: Initial z value
        :param vz0: Initial vz value
        :param tFin: Final time
        :param path: Path for the results file
        :return: Simulation results or the final iteration of the results, depending on returnTotal
        """

        # Read the results from the file
        t, z, vz = self.run(e=e, z0=z0, vz0=vz0, tFin=tFin, filename="", path=path, returnTotal=True, debug=False)

        # Check if the conditions are stable
        stable = self.checkIfStable(t, z, vz)

        # Count the number of crossings
        numCrossings = _crossings(t, z, vz)

        return e, z0, vz0, int(stable), len(numCrossings)

    def criteriaStable(self, t, z, vz):
        """
        Check stability criteria based on z, vz, and its derivative.
        Returns the index of the first instability or -1 if stable.
        """
        derivative = np.gradient(vz, t)
        mask = (
                ((z > self.zCriteria) & (vz > self.vzCriteria) & (derivative > -self.azCriteria)) |
                ((z < -self.zCriteria) & (vz < self.vzCriteria) & (derivative < self.azCriteria))
        )
        indices = np.where(mask)[0][::10]
        return indices[0] if indices.size else -1

    def checkIfStable(self, t, z, vz):
        """
        Check if the entire duration is stable.
        """
        return self.criteriaStable(t, z, vz) == -1

    def processFile(self, filename: str, e: float, z0: float, vz0: float) -> tuple:
        """
        Extracts and processes data from the specified file to determine initial conditions.

        :param filename: The path to the file containing the data.
        :param e: Parameter e's value.
        :param z0: Initial z value.
        :param vz0: Initial Vz value.
        :return: A tuple containing e, z0, Vz0, stability (as an int), and the number of crossings.
        """
        # Read the results from the file
        t, z, vz = readResults(filename)

        # Check if the conditions are stable
        stable = self.checkIfStable(t, z, vz)

        # Count the number of crossings
        numCrossings = _crossings(t, z, vz)

        return e, z0, vz0, int(stable), len(numCrossings)


def readResults(filename):
    """
    Reads the simulation results from a given filename.

    :param filename: Name of the file to read the results from.
    :return: Arrays of time (t), z values, and vz values.
    """
    with open(filename, "r") as f:
        lines = f.readlines()[1:]
        t, z, vz = zip(*[(np.float64(val[0]), np.float64(val[2]), np.float64(val[3]))
                         for val in (line.split() for line in lines) if val[0].replace('.', '', 1).isdigit()])
    return np.array([t, z, vz])


def _crossings(t, z, vz):
    """
    Identify the time indices where z is close to zero and the change rate of vz (az) is small.

    Parameters:
    - t: Time array
    - z: z-values corresponding to time t
    - vz: vz-values (velocity in z direction) corresponding to time t

    Returns:
    - crossingsVar: List of indices where the conditions are met
    """
    derivative = np.gradient(vz, t)
    zCriteria = 1e-1
    azCriteria = 5e-1
    difference = 0.5
    lastTimeCrossing = -1
    crossingsVar = []

    for i in range(0, len(t)):
        tempZ, tempAz = z[i], derivative[i]
        if abs(tempZ) < zCriteria and abs(tempAz) < azCriteria:
            if (t[i] - lastTimeCrossing) > difference:
                lastTimeCrossing = t[i]
                crossingsVar.append(i)

    return crossingsVar


if __name__ == '__main__':
    sitnikov = SitnikovProblem()
    sitnikov.run(e=0, z0=0, vz0=1, tFin=60, path="Simulations")
    print(readResults("Simulations/e=0_z0=0_vz0=1.txt").shape)
