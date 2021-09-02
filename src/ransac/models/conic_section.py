import ransac.core as ransac
import numpy as np
import cmath
import math

class ConicSection(ransac.Model):  # Input is (x, y) and output is Ax**2 + Bxy + Cy**2 + Dx + Ey + F.
    #                                A point belonging to the conic section will return 0
    def __init__(self):
        super().__init__()
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.F = 0

    def Evaluate(self, xy, zero_threshold=1.0E-15):  # Take an input variable x and return an output variable y
        x = xy[0]
        y = xy[1]
        return self.A * x ** 2 + self.B * x * y + self.C * y ** 2 + self.D * x + self.E * y + self.F

    def Distance(self, y1, y2):  # Compute the distance between two output variables. Must return a float.
        return abs(y1 - y2)

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        A = np.zeros((len(xy_tuples), 6), dtype=float)
        for obsNdx in range(len(xy_tuples)):
            ((x, y), d) = xy_tuples[obsNdx]
            A[obsNdx, 0] = x**2
            A[obsNdx, 1] = x * y
            A[obsNdx, 2] = y**2
            A[obsNdx, 3] = x
            A[obsNdx, 4] = y
            A[obsNdx, 5] = 1
        # Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
        # Find the eigenvalues and eigenvector of A^T A
        e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))

        # Extract the eigenvector (column) associated with the minimum eigenvalue
        z = e_vecs[:, np.argmin(e_vals)]
        self.A = z[0]
        self.B = z[1]
        self.C = z[2]
        self.D = z[3]
        self.E = z[4]
        self.F = z[5]

    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number or (x, y) observations to define the model
        return 6

    def ConicSectionType(self, zero_threshold=1.E-15):
        # Cf. https://www.varsitytutors.com/hotmath/hotmath_help/topics/conic-sections-and-standard-forms-of-equations
        gamma = self.B**2 - 4 * self.A * self.C
        if abs(gamma) <= zero_threshold:
            return 'parabola'
        if gamma < 0:
            return 'ellipse'
        else:
            return 'hyperbola'

    def Center(self, zero_threshold=1.0E-15):
        denominator = -self.B**2 + 4 * self.A * self.C
        if abs(denominator) < zero_threshold:
            raise ValueError("ConicSection.Center(): abs(-B^2 + 4AC) ({}) is below the zero threshold ({})".format(abs(denominator), zero_threshold))
        return ((self.B * self.E - 2 * self.C * self.D)/denominator, (self.B * self.D - 2 * self.A * self.E)/denominator)

    def EllipseParameters(self, zero_threshold=1.0E-15):
        conic_section_type = self.ConicSectionType(zero_threshold)
        if not conic_section_type == 'ellipse':
            raise ValueError("ConicSection.EllipseParameters(): The type of conic section ({}) is not 'ellipse'".format(conic_section_type))
        center = self.Center(zero_threshold)
        xc = center[0]
        yc = center[1]

        # Compute the ellipse with the same shape and orientation, but centered on (0, 0)
        # fc(x, y) = A(x+xc)**2 + B(x+xc)(y+yc) + C(y+yc)**2 + D(x+xc) + E(y+yc) + F = 0
        Ac = self.A;    Bc = self.B;    Cc = self.C
        Dc = 0  # 2 * self.A * xc + self.B * yc + self.D  (*)
        Ec = 0  # self.B * xc + 2 * self.C * yc + self.E  (*)
        Fc = self.A * xc**2 + self.B * xc * yc + self.C * yc**2 + self.D * xc + self.E * yc + self.F
        # (*) After substituting xc = (BE-2CD)/(-B**2+4AC) and yc = (BD-2AE)/(-B**2+4AC), we can verify that Dc=0 and Ec=0

        if abs(Fc) <= zero_threshold:
            raise ValueError("ConicSection.EllipseParameters(): abs(Fc) ({}) <= {}".format(abs(Fc), zero_threshold))
        USUT = np.array([[Ac, 0.5 * Bc], [0.5 * Bc, Cc]])
        U, s, UT = np.linalg.svd(USUT, full_matrices=True)

        theta = cmath.phase(complex(UT[0, 0], UT[0, 1]))
        sin2T = math.sin(theta)**2
        cos2T = math.cos(theta)**2
        A = np.zeros((2, 2), dtype=float)
        c = np.zeros(2, dtype=float)
        A[0, 0] = cos2T;    A[0, 1] = sin2T
        A[1, 0] = sin2T;    A[1, 1] = cos2T
        c[0] = -Ac/Fc
        c[1] = -Cc/Fc
        z = np.linalg.solve(A, c)

        if abs(z[0] <= zero_threshold):
            raise ValueError("ConicSection.EllipseParameters(): abs(z[0]) ({}) <= {}".format(abs(z[0]), zero_threshold))
        if abs(z[1] <= zero_threshold):
            raise ValueError("ConicSection.EllipseParameters(): abs(z[1) ({}) <= {}".format(abs(z[1]), zero_threshold))
        a = math.sqrt(1/z[0])
        b = math.sqrt(1/z[1])
        if b > a:
            temp_a = a
            a = b
            b = temp_a
            theta += math.pi/2
        if theta > math.pi/2:
            theta -= math.pi
        if theta < -math.pi/2:
            theta -= math.pi

        return center, a, b, theta

    def ClosestPoint(self, xy, zero_threshold=1.0E-15):
        center = self.Center(zero_threshold)
        centered_xy = (xy[0] - center[0], xy[1] - center[1])
        radial_points = []
        alphas = list(range(21))
        alphas = [0.1 * i for i in alphas]
        for alpha in alphas:
            p = (center[0] + alpha * centered_xy[0], center[1] + alpha * centered_xy[1])
            radial_points.append(p)
        with open('./outputs/ConicSection_ClosestPoint_radialPoints.csv', 'w+') as radial_points_file:
            radial_points_file.write("x,y,f(xy)\n")
            for radial_point in radial_points:
                f_xy = self.Evaluate(radial_point, zero_threshold)
                radial_points_file.write("{},{},{}\n".format(radial_point[0], radial_point[1], f_xy))

        return (0, 0)
