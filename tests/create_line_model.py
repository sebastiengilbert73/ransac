import logging
import ransac.core as ransac
import ransac.models.line as ransac_line


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("create_line_model.py main()")

    line = ransac_line.Line()
    xy_tuples = [((2, -3), 0), ((5, -3), 0), ((9.5, -3.3), 0)]
    line.Create(xy_tuples)
    logging.info("line.rho = {}; line.theta = {}".format(line.rho, line.theta))


if __name__ == '__main__':
    main()