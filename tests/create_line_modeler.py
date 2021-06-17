import logging
import ransac.core as ransac
import ransac.models.line as ransac_line
import random
import math
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("create_line_modeler.py main()")

    real_rho = 4.8
    real_theta = 1.2
    number_of_inliers = 200
    alpha_range = [-10, 10]  # The range of distances along the line, from (rho * cos(theta), rho * sin(theta))
    inliers_noise = 0.2
    number_of_outliers = 50
    number_of_trials = 100
    acceptable_error = 0.4

    xy_tuples = []
    for inlierNdx in range(number_of_inliers):
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        pt = [real_rho * math.cos(real_theta) + alpha * math.sin(real_theta),
              real_rho * math.sin(real_theta) - alpha * math.cos(real_theta)]
        pt[0] += random.uniform(-inliers_noise, inliers_noise)
        pt[1] += random.uniform(-inliers_noise, inliers_noise)
        xy_tuples.append((pt, 0))
    for outlierNdx in range(number_of_outliers):
        pt = [random.uniform(alpha_range[0], alpha_range[1]), random.uniform(alpha_range[0], alpha_range[1])]
        xy_tuples.append((pt, 0))

    line_modeler = ransac.Modeler(ransac_line.Line, number_of_trials, acceptable_error)
    consensus_model, inliers_list, outliers_list = line_modeler.ConsensusModel(xy_tuples)
    logging.info("consensus_model: rho = {}; theta = {}".format(consensus_model.rho, consensus_model.theta))

    real_line_list = [(real_rho * math.cos(real_theta) + alpha * math.sin(real_theta),
              real_rho * math.sin(real_theta) - alpha * math.cos(real_theta))
                       for alpha in [x/10 for x in range(10 * int(alpha_range[0]), 10 * int(alpha_range[1]))] ]

    found_line_list = [(consensus_model.rho * math.cos(consensus_model.theta) + alpha * math.sin(consensus_model.theta),
                        consensus_model.rho * math.sin(consensus_model.theta) - alpha * math.cos(consensus_model.theta))
                      for alpha in [x / 10 for x in range(10 * int(alpha_range[0]), 10 * int(alpha_range[1]))]]


    # Display the results
    fig, ax = plt.subplots()
    ax.scatter([inlier[0][0] for inlier in inliers_list], [inlier[0][1] for inlier in inliers_list],
               c='green', label='inliers')
    ax.scatter([outlier[0][0] for outlier in outliers_list], [outlier[0][1] for outlier in outliers_list],
               c='red', label='outliers')
    ax.scatter([p[0] for p in real_line_list], [p[1] for p in real_line_list],
               c='blue', label='real line', s=1)
    ax.scatter([p[0] for p in found_line_list], [p[1] for p in found_line_list],
               c='fuchsia', label='consensus line', s=1)
    ax.legend()
    ax.grid(True)
    plt.show()



if __name__ == '__main__':
    main()