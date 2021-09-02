import logging
import ransac.core as ransac
import ransac.models.polynomial as polynomial_model
import random
import math
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("create_polynomial_modeler.py main()")

    real_c0 = -3.4
    real_c1 = 0.8
    real_c2 = -1.5
    real_c3 = 0.3
    number_of_trials = 200
    acceptable_error = 1.0

    number_of_inliers = 100
    number_of_outliers = 100
    noise_amplitude = 0.5

    xy_tuples = []
    xs = np.random.random(number_of_inliers).tolist()
    xs = [-3 + 6 * x for x in xs]  # Scale the range
    for x in xs:
        y = real_c0 + real_c1 * x + real_c2 * x**2 + real_c3 * x**3
        noise = noise_amplitude * (-1 + 2 * np.random.random())
        xy_tuples.append((x, y + noise))
    outliers = []
    for outlierNdx in range(number_of_outliers):
        x = -3 + 6 * np.random.random()
        y = -25 + 30 * np.random.random()
        xy_tuples.append((x, y))
    # Shuffle the observations
    np.random.shuffle(xy_tuples)

    logging.debug("len(xy_tuples) = {}".format(len(xy_tuples)))
    # Create the polynomial modeler
    polynomial_modeler = ransac.Modeler(polynomial_model.Polynomial, number_of_trials, acceptable_error)
    consensus_polynomial, inliers_list, outliers_list = polynomial_modeler.ConsensusModel(xy_tuples,
                                                                                          degree=3)

    logging.info("consensus_polynomial.coefficients = {}".format(consensus_polynomial.coefficients))

    # Display the results
    fig, ax = plt.subplots()
    ax.scatter([inlier[0] for inlier in inliers_list], [inlier[1] for inlier in inliers_list],
               c='green', label='inliers')
    ax.scatter([outlier[0] for outlier in outliers_list], [outlier[1] for outlier in outliers_list],
               c='red', label='outliers')

    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()