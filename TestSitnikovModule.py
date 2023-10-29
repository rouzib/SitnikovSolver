import time
import numpy as np
from matplotlib import pyplot as plt

import SitnikovModule


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


class TestSitnikovProblem:
    def __init__(self):
        """
        Initialize the problem constants and create an instance of the Sitnikov problem solver.
        """
        self.zCriteria = 10
        self.vzCriteria = 0.5
        self.azCriteria = 1e-2
        self.sitnikovProblem = SitnikovModule.SitnikovProblem()

    def runAndPlot(self, e, z0, vz0, tFin, valuesToShow=None):
        """
        Run the Sitnikov problem for given parameters and plot the results.
        """
        if valuesToShow is None:
            valuesToShow = [0, 1, 2]
        self.plotAll(*self.sitnikovProblem.run(e, z0, vz0, tFin, "test.txt", "", returnTotal=True), valuesToShow,
                     parameters=[e, z0, vz0])

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

    def plotAll(self, t, z, vz, valuesToShow=None, parameters=None):
        """
        Plot the results: z(t), vz(t), and its derivative.
        """

        def plotStableOrUnstable(t, z, vz):
            i = self.criteriaStable(t, z, vz)
            if i != -1:
                plt.axvline(t[i], color="k")
                return False
            return True

        derivative = np.gradient(vz, t)
        if valuesToShow is None:
            valuesToShow = [0]

        # Plotting z, vz, and az based on the user's selection
        if 0 in valuesToShow:
            plt.plot(t, z, label="z")
        if 1 in valuesToShow:
            plt.plot(t, vz, label="vz")
        if 2 in valuesToShow:
            plt.plot(t, derivative, label="az")

        # Check for stability
        stable = plotStableOrUnstable(t, z, vz)

        # Plot the crossings
        crossings = _crossings(t, z, vz)
        plt.scatter(t[crossings], z[crossings], c="red", s=5, zorder=10)

        # Set titles and labels
        plt.title(
            f"Orbit of the third mass (number of crossings: {len(crossings)})"
            f"\n{'e={:.4f}, z0={:.4f}, vz0={:.4f}'.format(*parameters) if parameters else ''}"
            f"\nStable = {stable}"
        )
        plt.xlabel(r"T ($2\pi$)")
        plt.ylabel(f"z(t){' / vz(t)' if 1 in valuesToShow else ''}{' / az(t)' if 2 in valuesToShow else ''}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    e, z0, vz0 = 0.5, 0.5930655598640442, 0.41275277733802795
    tFin = 70

    sitnikov = TestSitnikovProblem()

    start_time = time.time()  # Start the timer

    sitnikov.runAndPlot(e, z0, vz0, tFin, [0])

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
