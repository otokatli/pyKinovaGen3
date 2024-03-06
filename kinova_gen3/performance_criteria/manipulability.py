"""Manipulability objective function and its gradient

Functions
---------
manipulability(joint_position)
manipulability_gradient(joint_position)

"""

import numpy as np
from numpy.linalg import det
from scipy.optimize import approx_fprime
from kinova_gen3.kinematics.jacobian import jacobian


def manipulability(joint_position):
    """Manipulability objective function

    Calculate the cost of the manipulatility for a given joint configuration

    Arguments
    ---------
    joint_position (array_like): Joint positions of the robot [rad]

    Returns
    -------
    float: Manipulability of the robot [non-dimensional]

    """

    return np.log(det(jacobian(joint_position) @ jacobian(joint_position).transpose()))


def manipulability_gradient(joint_position):
    """Gradient of the manipulability objective function

    Arguments
    ---------
    joint_position (array_like): Joint positions of the robot [rad]

    Returns
    -------
    ndarray: Gradient of the manipulability of the robot [non-dimensional]

    """

    return approx_fprime(joint_position, manipulability, np.sqrt(np.finfo(float).eps))
