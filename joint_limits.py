import numpy as np
from scipy.optimize import approx_fprime


def joint_limits(q):
    """ Joint limits objective function

    Calculate the cost of the current joint configuration from the lower
    and upper bounds of the joints

    :param q: Joint position of the robot
    type q: numpy.ndarray
    :return: Cost of the current configuration
    :type: float
    """
    
    q_min = np.array([-180.0, -128.9, -180.0, -147.8, -180.0, -120.3, -180.0]) * np.pi / 180.0
    q_max = np.array([180.0, 128.9, 180.0, 147.8, 180.0, 120.3, 180.0]) * np.pi / 180.0

    q_bar = (q_min + q_max) / 2

    return (-1 / (2 * q.shape[0])) * np.sum(((q - q_bar) / (q_max - q_min)) ** 2)


def joint_limits_gradient(q):
    """ Gradient of the joint limits objective function

    Calculate the first derivative of the joint limits objective
    function with respect to all joint variables

    :param q: Joint position of the robot
    :type q: numpy.ndarray
    :return: Gradient vector for the joint limits objective function
    :type: numpy.ndarray
    """

    return approx_fprime(q, joint_limits, np.sqrt(np.finfo(float).eps)) 
