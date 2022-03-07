import logging
import math
from typing import Tuple, Union

import numpy as np
np.seterr(all='ignore', divide='ignore', invalid='ignore')

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

numeric = Union[int, float, complex]

logger = logging.getLogger(__name__)


def custom_round(x: numeric):
    """Implements a custom round function for consistency purposes between
    python 2 and python 3. Python 2 "rounds to nearest, ties away from zero" and
    Python 3 uses "round to nearest, ties to nearest even".

    :param x: The number to round.
    :type x: float
    :return: The rounded number.
    :rtype: float
    """
    out = math.floor(x)
    if x < 0:
        return out if x - out < 0.5 else out
    else:
        return out if x - out < 0.5 else out + 1


def matlab_percentile(in_data, percentiles):
    """Source: https://github.com/numpy/numpy/issues/6620

    Calculate percentiles in the way IDL and Matlab do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    :param in_data: Input data.
    :type in_data: numpy.ndarray
    :param percentiles: Values of the percentiles.
    :type percentiles: numpy.ndarray
    """
    data = np.sort(in_data)
    p_rank = 100 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def sigmoid(x_val, para1, para2, para3, para4):
    """
    Given an array of values (x) and parameters p1, p2, p3 and p4 sigmoid
    evaluates the function:
    y = p1 + (p2-p1) / (1.+10.**((p3-x)*p4))
    where y is an array of the same size as x.

    Used as the function definition for curve_fit within sigm_fit or for
    evaluating the sigmoid for given x values.

    :param x_val: one dimensional array of dtype float64
    :param para1: float64 scalar, denoting the minimum value of the sigmoid
    :param para2: float64 scalar, denoting the maximum value of the sigmoid
    :param para3: float64 scalar, denoting the x value of the half-max of the
                  sigmoid
    :param para4: float64 scalar, denoting the slope of the sigmoid
    :return: one dimensional array of dtype float64, same size as x
    """
    return para1 + (para2 - para1) / (1 + 10.0 ** ((para3 - x_val) * para4))


def sigm_fit(xval1, yval1):
    """
    This function takes arrays for the x and y coordinates of a set of points
    and fits a sigmoid function to the points using a nonlinear least squares 
    regression within scipy.optimize.curve_fit. In the case that the 
    least-squares minimization fails, the function returns a list of four NaNs.

    :param xval1: one dimensional array of dtype float64
    :param yval1: one dimensional array of dtype float64, same size as x1
    :return: 1x4 array of dtype float64 containing the optimized parameters
             p1, p2, p3 and p4 as in sigmoid
    """
    params, temp = None, None
    if (len(yval1) > 0) and (len(xval1) > 0) and (len(yval1) == len(xval1)):
        if not [k for k in yval1 if k == matlab_percentile(yval1, 50.0)]:
            idx = [k for k in range(len(yval1[1:])) if yval1[k] == matlab_percentile(yval1[1:], 50.0)]
            if idx:
                temp = xval1[idx]
        else:
            idx = [k for k in range(len(yval1)) if yval1[k] == matlab_percentile(yval1, 50.0)]
            if idx:
                temp = xval1[idx]

        if temp is not None:
            try:
                params_init = [matlab_percentile(yval1, 5.0), matlab_percentile(yval1, 95.0), temp[0], 1.0]
                params = curve_fit(sigmoid, xval1, yval1, p0=params_init)[0]
            except Exception as error:
                logging.debug("Error fitting sigmoid curve: %s", error)

    return params


def compute_difference(image_arr, difference1, standard_dev):
    """[summary]"""

    if difference1 is not None:
        difference = difference1 * (1 - 2.0 * standard_dev)
    else:
        set_interp = interp1d([8, 12, 16], [8, 12, 16], kind='linear')
        x_new = [1 if np.max(image_arr) == 0 else math.ceil(math.log(np.max(image_arr), 2))]
        difference = 2 ** set_interp(x_new)[0] / 8 * (1 - 2.0 * standard_dev)

    return difference


def compute_theta_and_rho(p1: Tuple[int], p2: Tuple[int]):
    """
    This function takes two points p1 and p2 and computes theta, which is the angle 
    between the positive x-axis and the edge, and rho, which is the distance between 
    the edge and the origin. 

    :param p1: a tuple containing the image coordinates of the first point.
    :type p1: Tuple[int]
    :param p2: a tuple containing the image coordinates of the second point. 
    :type p2: Tuple[int]
    :return: theta and rho.
    :rtype: float, float
    """

    # Get the slope of the the edge
    try:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    except ZeroDivisionError:
        slope = float('Inf')

    intcpt = p2[1] - slope * p2[0]

    # Compute Theta and Rho
    try:
        theta = float(math.atan(-1 / slope))
    except ZeroDivisionError:
        theta = float(-math.pi / 2)
    rho = float(intcpt * math.sin(theta))

    return theta, rho
