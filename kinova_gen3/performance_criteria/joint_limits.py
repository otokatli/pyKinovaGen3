"""Joint limit objective function and its gradient

Functions
---------
joint_limits(joint_position)
joint_limits_gradient(joint_position)

"""

import numpy as np
from scipy.optimize import approx_fprime


def joint_limits(joint_position):
    """Joint limits objective function

    Calculate the cost of the current joint configuration from the lower
    and upper bounds of the joints

    Arguments
    ---------
    joint_position (array_like): Joint position of the robot [rad]

    Returns
    -------
    float: Cost of the current configuration [non-dimensional]

    """

    joint_position_min = (
        np.array([-180.0, -128.9, -180.0, -147.8, -180.0, -120.3, -180.0])
        * np.pi
        / 180.0
    )
    joint_position_max = (
        np.array([180.0, 128.9, 180.0, 147.8, 180.0, 120.3, 180.0]) * np.pi / 180.0
    )

    joint_position_bar = (joint_position_min + joint_position_max) / 2

    return (-1 / (2 * joint_position.shape[0])) * np.sum(
        (
            (joint_position - joint_position_bar)
            / (joint_position_max - joint_position_min)
        )
        ** 2
    )


def joint_limits_gradient(joint_position):
    """Gradient of the joint limits objective function

    Calculate the first derivative of the joint limits objective
    function with respect to all joint variables

    Arguments
    ---------
    joint_position (array_like): Joint position of the robot [rad]

    Returns
    -------
    ndarray: Gradient vector for the joint limits objective function [non-dimensional]

    """

    return approx_fprime(joint_position, joint_limits, np.sqrt(np.finfo(float).eps))
