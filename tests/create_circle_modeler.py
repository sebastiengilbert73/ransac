import logging
import ransac.core as ransac
import ransac.models.circle as circle_model
import random
import math
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("create_circle_modeler.py main()")

    real_center = (5, 3)
    real_radius = 4.5
    number_of_inliers = 200
    inliers_noise = 0.5
    number_of_outliers = 50
    outliers_range = [-12, 12]
    number_of_trials = 100
    acceptable_error = 0.7

    xy_tuples = []
    for inlierNdx in range(number_of_inliers):
        theta = random.uniform(0, 2 * math.pi)
        p = [real_center[0] + real_radius * math.cos(theta),
             real_center[1] + real_radius * math.sin(theta)]
        p[0] += random.uniform(-inliers_noise, inliers_noise)
        p[1] += random.uniform(-inliers_noise, inliers_noise)
        xy_tuples.append((p, 0))

    for outlierNdx in range(number_of_outliers):
        p = (random.uniform(outliers_range[0], outliers_range[1]),
             random.uniform(outliers_range[0], outliers_range[1]))
        xy_tuples.append((p, 0))

    # Create the circle modeler
    circle_modeler = ransac.Modeler(circle_model.Circle, number_of_trials, acceptable_error)
    consensus_circle, inliers_list, outliers_list = circle_modeler.ConsensusModel(xy_tuples)
    logging.info("consensus_circle.center = {}; consensus_circle.radius = {}".format(consensus_circle.center, consensus_circle.radius))

    real_circle_list = [(real_center[0] + real_radius * math.cos(theta),
                         real_center[1] + real_radius * math.sin(theta))
                         for theta in np.arange(0, 2 * math.pi, 0.01)]
    found_circle_list = [(consensus_circle.center[0] + consensus_circle.radius * math.cos(theta),
                          consensus_circle.center[1] + consensus_circle.radius * math.sin(theta))
                         for theta in np.arange(0, 2 * math.pi, 0.01)]

    # Display the results
    fig, ax = plt.subplots()
    ax.scatter([inlier[0][0] for inlier in inliers_list], [inlier[0][1] for inlier in inliers_list],
               c='green', label='inliers')
    ax.scatter([outlier[0][0] for outlier in outliers_list], [outlier[0][1] for outlier in outliers_list],
               c='red', label='outliers')
    ax.scatter([p[0] for p in real_circle_list], [p[1] for p in real_circle_list],
               c='blue', label='real circle', s=1)
    ax.scatter([p[0] for p in found_circle_list], [p[1] for p in found_circle_list],
               c='fuchsia', label='consensus circle', s=1)
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    main()