"""Inverse kinematics module for Kinova Gen3 robot

Functions
---------
inverse_kinematics(joint_position, end_effector_vel)
multicriteria_ik(joint_position, end_effector_vel)
inverse_kinematics_dls(joint_position, end_effector_vel, k)
multicriteria_ik_damped(joint_position, end_effector_vel)

"""

import numpy as np
from .jacobian import jacobian
from performance_criteria.manipulability import manipulability_gradient
from performance_criteria.joint_limits import joint_limits_gradient


def inverse_kinematics(joint_position, end_effector_vel):
    """Velocity level inverse kinematics of the robot

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot [rad]
    end_effector_vel (array_like): The end-effector velocities of the robot [m/s]

    Returns
    -------
    ndarray: The joint velocities [rad/s]

    """

    return np.linalg.pinv(jacobian(joint_position)) @ end_effector_vel


def _nullspace_projector(joint_position):
    """The projector for mapping an arbitrary vector to the nullspace of the Jacobian

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot [rad]

    Returns
    -------
    ndarray: The nullspace projector (I - pinv(J) J)

    """

    return np.eye(7) - np.linalg.pinv(jacobian(joint_position)) @ jacobian(
        joint_position
    )


def multicriteria_ik(joint_position, end_effector_vel):
    """Multicriteria Inverse Kinematics algorithm

    Solve the velocity level inverse kinematics problem while optimising the
    manipulability of the robot and avoiding the physical joint limits.

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot [rad]
    end_effector_vel (array_like): End-effector velocity [m/s]

    Returns
    -------
    ndarray: Joint velocities [rad/s]

    """

    z_manipulability = manipulability_gradient(joint_position)
    z_joint_limits = joint_limits_gradient(joint_position)

    # The weight of the manipulability objective function
    # Note that the sum of all weights should be ejoint_positionual to 1
    alpha = 0.75

    # A positive constant for the nullspace projection
    k_z = 5

    z_arbitrary_vel = alpha * z_manipulability + (1 - alpha) * z_joint_limits

    return (
        inverse_kinematics(joint_position, end_effector_vel)
        + k_z * _nullspace_projector(joint_position) @ z_arbitrary_vel
    )


def inverse_kinematics_dls(joint_position, end_effector_vel, k=0.01):
    """Damped least squares solution for the velocity-level inverse kinematics

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot [rad]
    end_effector_vel (array_like): End-effector velocity [m/s]
    k (float): Damping for the least-sjoint_positionuares [non-dimensional]

    Returns
    -------
    ndarray: Joint velocities [rad/s]

    """

    return (
        jacobian(joint_position).transpose()
        @ np.linalg.inv(
            jacobian(joint_position) @ jacobian(joint_position).transpose()
            + k**2 * np.eye(6)
        )
        @ end_effector_vel
    )


def multicriteria_ik_damped(joint_position, end_effector_vel):
    """Multicriteria Inverse Kinematics algorithm with damped least-sjoint_positionuares

    Solve the velocity level inverse kinematics problem while optimising the
    manipulability of the robot and avoiding the physical joint limits. This
    implementation uses damped least-sjoint_positionuares for inverting the Jacobian matrix.
    Damped least sjoint_positionuares methods should provide better results in the viscinity
    of singularities.

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot [rad]
    end_effector_vel (array_like): End-effector velocity [m/s]

    Returns
    -------
    ndarray: Joint velocities [rad/s]

    """

    z_manipulability = manipulability_gradient(joint_position)
    z_joint_limits = joint_limits_gradient(joint_position)

    # The weight of the manipulability objective function
    # Note that the sum of all weights should be ejoint_positionual to 1
    alpha = 0.75

    # A positive constant for the nullspace projection
    k_z = 5

    z_arbitrary_vel = alpha * z_manipulability + (1 - alpha) * z_joint_limits

    return (
        inverse_kinematics_dls(joint_position, end_effector_vel)
        + k_z * _nullspace_projector(joint_position) @ z_arbitrary_vel
    )
