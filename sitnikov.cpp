#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <array>
#include <functional>
#include <chrono>


// Coefficients for the RKF 7(8) method
const std::array<double, 13> a = {0.0, 2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0, 0.5, 5.0 / 6.0, 1.0 / 6.0,
                                  2.0 / 3.0,
                                  1.0 / 3.0, 1.0, 0.0, 1.0};
const std::array<std::array<double, 13>, 13> b = {{
                                                          {0.0},
                                                          {2.0 / 27.0, 0.0},
                                                          {1.0 / 36.0, 3.0 / 36.0, 0.0},
                                                          {1.0 / 24.0, 0.0, 3.0 / 24.0, 0.0},
                                                          {20.0 / 48.0, 0.0, -75.0 / 48.0, 75.0 / 48.0, 0.0},
                                                          {1.0 / 20.0, 0.0, 0.0, 5.0 / 20.0, 4.0 / 20.0, 0.0},
                                                          {-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -260.0 / 108.0,
                                                           250.0 / 108.0, 0.0},
                                                          {31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0,
                                                           13.0 / 900.0, 0.0},
                                                          {2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0,
                                                           67.0 / 90.0, 3.0, 0.0},
                                                          {-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0,
                                                           311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0, -1.0 / 12.0, 0.0},
                                                          {2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                                                           -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0,
                                                           18.0 / 41.0, 0.0},
                                                          {3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0,
                                                           -3.0 / 41.0, 3.0 / 41.0, 6.0 / 41.0, 0.0, 0.0},
                                                          {-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                                                           -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0,
                                                           12.0 / 41.0, 1.0, 0.0}
                                                  }};

const std::array<double, 13> c = {41.0 / 840.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0,
                                  9.0 / 280.0, 41.0 / 840.0, 0.0, 0.0};
const std::array<double, 13> d = {0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0,
                                  9.0 / 280.0,
                                  0.0, 41.0 / 840.0, 41.0 / 840.0};

/**
 * @brief System of ODEs representing the Sitnikov problem.
 *
 * This function computes the derivatives of z, vz, and theta.
 *
 * @param t Current time.
 * @param y Current state [z, vz, theta].
 * @param e eccentricity of the primary bodies
 * @return Vector of derivatives [dz/dt, dvz/dt, dtheta/dt].
 */
std::vector<double> sitnikovODEs(double t, const std::vector<double> &y, double e) {
    double dzdt = y[1];
    double primaryDistance = ((-e * e + 1.0) / (e * cos(y[2]) + 1.0));
    double dvzdt =  -y[0] / pow(primaryDistance * primaryDistance / 4.0 + y[0] * y[0], 1.5);
    double dthetadt = (pow(e * cos(y[2]) + 1.0, 2.0) / pow(-pow(e, 2.0) + 1.0, 1.5));

    return {dzdt, dvzdt, dthetadt};
}


/**
 * @brief Runge-Kutta-Fehlberg 7(8) method for solving ODEs.
 *
 * @param t0 Initial time.
 * @param t_final Final time.
 * @param y0 Initial state.
 * @param h_initial Initial step size.
 * @param tolerance The error tolerance for adaptive step size adjustment.
 * @param f Function pointer to the system of ODEs.
 * @param filename The name of the file to output to.
 * @param stopCriterion Function to evaluate stopping conditions based on y and t.
 * @return Final state after integrating up to t_final.
 */std::vector<double> RKF78_Solver(double t0, double t_final, std::vector<double> &&y0, double h_initial,
                                    double tolerance,
                                    const std::function<std::vector<double>(double, const std::vector<double> &)> &f,
                                    const char *filename,
                                    const std::function<bool(const std::vector<double> &, const std::vector<double> &,
                                                             double, double)> &stopCriterion) {
    double t = t0;
    double t_prev;
    std::vector<double> y = y0;
    std::vector<double> y_prev;
    double h = h_initial;

    std::ofstream outfile;  // File stream

    // Open the file (std::ios::app for append mode)
    if (filename && *filename) {
        outfile.open(filename, std::ios::out);
    }

    // Iteratively solve the ODE using the RKF 7(8) method
    while (t < t_final) {
        // std::cout << t << " / " << t_final << " " << h << std::endl;

        y_prev = y;
        t_prev = t;

        // Compute intermediate stages k_i based on the ODE system and coefficients
        std::vector<std::vector<double>> k(y.size(), std::vector<double>(a.size(), 0.0));
        for (size_t i = 0; i < a.size(); ++i) {
            std::vector<double> y_temp = y;
            for (size_t j = 0; j < i; ++j) {
                for (size_t l = 0; l < y.size(); ++l) {
                    y_temp[l] += h * b[i][j] * k[l][j];
                }
            }
            std::vector<double> dy = f(t + a[i] * h, y_temp);
            for (size_t l = 0; l < y.size(); ++l) {
                k[l][i] = dy[l];
            }
        }

        // Compute the 7th order solution estimate
        std::vector<double> y_new7 = y;
        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < a.size(); ++j) {
                y_new7[i] += h * c[j] * k[i][j];
            }
        }

        // Compute the 8th order solution estimate
        std::vector<double> y_new8 = y;
        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < a.size(); ++j) {
                y_new8[i] += h * d[j] * k[i][j];
            }
        }

        // Estimate the error between the 7th and 8th order solutions
        double error = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            error += std::pow(y_new7[i] - y_new8[i], 2);
        }
        error = std::sqrt(error);

        // Adapt the step size based on the error
        if (error < tolerance) {
            t += h;
            y = y_new7;
            h *= 1.1;  // Increase step size if error is small

            // If file is opened successfully, write the values to the file
            if (outfile.is_open()) {
                outfile << t << " " << h << " " << y[0] << " " << y[1] << " " << error << std::endl;
            }


            if (stopCriterion(y, y_prev, t, t_prev)) {
                // If file is opened successfully, write the values to the file
                if (outfile.is_open()) {
                    outfile << "Stopped because of the criteria." << std::endl;
                }
                break;  // Stop the integration if the criterion is met
            }

        } else {
            h *= 0.5;  // Decrease step size if error is too large
        }
    }

    // Close the file if it was opened
    if (outfile.is_open()) {
        outfile.close();
    }

    return y;
}

/**
 * @brief Runge-Kutta-Fehlberg 7(8) method for solving ODEs.
 *
 * @param t0 Initial time.
 * @param t_final Final time.
 * @param y0 Initial state.
 * @param h_initial Initial step size.
 * @param tolerance The error tolerance for adaptive step size adjustment.
 * @param f Function pointer to the system of ODEs.
 * @param filename The name of the file to output to.
 * @return Final state after integrating up to t_final.
 */
std::vector<double>
RKF78_Solver(double t0, double t_final, std::vector<double> &&y0, double h_initial, double tolerance,
             std::vector<double> (*f)(double, const std::vector<double> &), const char *filename) {
    return RKF78_Solver(t0, t_final, std::move(y0), h_initial, tolerance, f, filename,
                        [](const std::vector<double> &, const std::vector<double> &, double, double) { return false; });
}

/**
 * @brief Runge-Kutta-Fehlberg 7(8) method for solving ODEs.
 *
 * @param t0 Initial time.
 * @param t_final Final time.
 * @param y0 Initial state.
 * @param h_initial Initial step size.
 * @param tolerance The error tolerance for adaptive step size adjustment.
 * @param f Function pointer to the system of ODEs.
 * @return Final state after integrating up to t_final.
 */
std::vector<double>
RKF78_Solver(double t0, double t_final, std::vector<double> &&y0, double h_initial, double tolerance,
             std::vector<double> (*f)(double, const std::vector<double> &)) {
    return RKF78_Solver(t0, t_final, std::move(y0), h_initial, tolerance, f, "");

}


bool sitnikovSimpleCriterion(const std::vector<double> &y, const std::vector<double> &y_prev, double t, double t_prev,
                             double zCriteria, double vzCriteria, double azCriteria) {

    double zSitnikov = y[0];
    double vzSitnikov = y[1];
    double azSitnikov = (y[1] - y_prev[1]) / (t - t_prev);

    if ((zSitnikov > zCriteria && vzSitnikov > vzCriteria && azSitnikov > -azCriteria) ||
        (zSitnikov < -zCriteria && vzSitnikov < vzCriteria && azSitnikov < azCriteria)) {
        return true;
    }

    return false;
}


/**
 * @brief Function to set up the Sitnikov problem and run the RKF 7(8) solver.
 *
 * @param t0 Initial time (default: 0.0).
 * @param t_final Final time (default: 60.0).
 * @param e Eccentricity parameter (default: 0.0).
 * @param y0 Initial state vector [z, vz, theta] (default: {0.0, 1.0, 0.0}).
 * @param h_initial Initial step size for the RKF 7(8) solver (default: 0.01).
 * @param tolerance Error tolerance for the RKF 7(8) solver (default: 1e-8).
 * @param zCriteria Stopping criteria based on z value (default: 10).
 * @param vzCriteria Stopping criteria based on vz value (default: 0.5).
 * @param azCriteria Stopping criteria based on az value (default: 0.01).
 * @param filename The name of the file to output to (default: "output.txt").
 * @param useCriterion If the criterion should be used (default: true).
 * @return Final state vector after integrating up to t_final.
 */
std::vector<double> runSitnikovSolver(double t0 = 0.0, double t_final = 60.0, double e = 0.0,
                                      std::vector<double> y0 = {0.0, 1.0, 0.0}, double h_initial = 0.01,
                                      double tolerance = 1e-8, double zCriteria = 10, double vzCriteria = 0.5,
                                      double azCriteria = 0.01, const char *filename = "output.txt",
                                      bool useCriterion = true) {

    auto criterion = [&zCriteria, &vzCriteria, &azCriteria](const std::vector<double> &y,
                                                            const std::vector<double> &y_prev,
                                                            double t, double prevT) {
        return sitnikovSimpleCriterion(y, y_prev, t, prevT, zCriteria, vzCriteria, azCriteria);
    };

    auto sitnikovODE = [e](double t, const std::vector<double> &y) {
        return sitnikovODEs(t, y, e);
    };

    // Solve the Sitnikov problem using the RKF 7(8) method
    std::vector<double> result = RKF78_Solver(t0, t_final, std::move(y0), h_initial, tolerance, sitnikovODE,
                                              filename, criterion);

    return result;
}

#if !BUILD_PYTHON_MODULE

/**
 * @brief Main function to run the solver and display results.
 */
int main() {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Solve the Sitnikov problem using the RKF 7(8) method
    std::vector<double> result = runSitnikovSolver(0, 100);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    // Compute the difference between the end time and start time
    std::chrono::duration<double> elapsed = finish - start;
    // Print elapsed time in seconds
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";


    // Display the results
    std::cout << "Final values: z = " << result[0] << ", vz = " << result[1] << ", theta = " << result[2] << std::endl;
    return 0;
}

#endif

