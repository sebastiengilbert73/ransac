import ransac.core as ransac
import math
import numpy as np
import logging


class Line(ransac.Model):  # The input is a 2D point. The output is the distance from the line
    def __init__(self, rho=None, theta=None, zero_threshold=1e-6):
        super().__init__()
        self.rho = rho
        self.theta = theta
        self.zero_threshold = zero_threshold

    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        # rho_prime: Distance of point (x0, x1) from the origin
        rho_prime = x[0] * math.cos(self.theta) + x[1] * math.sin(self.theta)
        # | self.rho - rho_prime | is the distance of the point to the line
        return abs(self.rho - rho_prime)

    def Distance(self, y1, y2):  # Compute the distance between two output variables
        return abs(y1 - y2)

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x0, x1) observations
        if len(xy_tuples) < 2:
            raise ValueError("line.Line.Create(): The number of points ({}) is less than 2".format(len(xy_tuples)))
        if 'zero_threshold' in kwargs:
            self.zero_threshold = kwargs['zero_threshold']
        if len(xy_tuples) == 2:  # We use only the point coordinates, assuming the corresponding distances to the line are 0
            x1 = xy_tuples[0][0]
            x2 = xy_tuples[1][0]
            if abs(x1[0] - x2[0]) <= self.zero_threshold and \
                    abs(x1[1] - x2[1]) <= self.zero_threshold:
                raise ValueError("line.Line.Create(): The two points coincide at ({}, {})".format(x1[0], x1[1]))
            if abs(x1[0] - x2[0]) <= self.zero_threshold:  # Vertical line
                self.rho = x1[0]
                self.theta = 0
            elif abs(x1[1] - x2[1]) <= self.zero_threshold:  # Horizontal line
                self.rho = x1[1]
                self.theta = math.pi/2
            else:  # The line is not vertical nor horizontal
                # x_i * cos(theta) + y_i * sin(theta) = rho
                # x_i + y_i * tan(theta) = rho/cos(theta)  Since we know cos(theta) != 0
                # | y_i  -1 | |   tan(theta)   |  =  | -x_i |
                # | ...  ...| | rho/cos(theta) |     |  ... |
                A = np.zeros((2, 2), dtype=float)
                b = np.zeros((2, 1), dtype=float)
                A[0, 0] = x1[1]
                A[0, 1] = -1
                A[1, 0] = x2[1]
                A[1, 1] = -1
                b[0, 0] = -x1[0]
                b[1, 0] = -x2[0]
                z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
                self.theta = math.atan(z[0, 0])
                self.rho = z[1, 0] * math.cos(self.theta)
        else:  # More than 2 points
            xs = [xy[0][0] for xy in xy_tuples]
            min_x = min(xs)
            max_x = max(xs)
            #logging.debug("line.Line.Create(): min_x = {}; max_x = {}".format(min_x, max_x))
            if abs(min_x - max_x) <= self.zero_threshold:  # Vertical line
                self.rho = min_x
                self.theta = 0
            else:  # Non-vertical line
                ys = [xy[0][1] for xy in xy_tuples]
                min_y = min(ys)
                max_y = max(ys)
                #logging.debug("line.Line.Create(): min_y = {}; max_y = {}".format(min_y, max_y))
                if abs(min_y - max_y) <= self.zero_threshold:  # Horizontal line
                    self.rho = min_y
                    self.theta = math.pi/2
                else:  # Non-horizontal line
                    # | x_i    y_i   -1  | | cos(theta) | = | 0 |
                    # | ...    ...   ... | | sin(theta) | = | 0 |
                    # | ...    ...   ... | |    rho     |   |...|
                    A = np.zeros((len(xy_tuples), 3), dtype=float)
                    for row in range(len(xy_tuples)):
                        x_i = xy_tuples[row][0][0]
                        y_i = xy_tuples[row][0][1]
                        A[row, 0] = x_i
                        A[row, 1] = y_i
                        A[row, 2] = -1
                    # Solution to a system of homogeneous linear equations. Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
                    e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
                    # Extract the eigenvector (column) associated with the minimum eigenvalue
                    z = e_vecs[:, np.argmin(e_vals)]
                    # Multiply by a factor such that cos^2(theta) + sin^2(theta) = 1
                    r2 = z[0]**2 + z[1]**2
                    if abs(r2) < self.zero_threshold:
                        raise ValueError(f"Line.Create(): z[0]**2 + z[1]**2 ({r2}) < {self.zero_threshold}")
                    z = z/math.sqrt(r2)
                    self.theta = np.angle(complex(z[0], z[1]))
                    self.rho = z[2]


    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number or (x, y) observations to define the model
        return 2  # Two points are necessary to define a line

