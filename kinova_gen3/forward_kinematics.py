"""Position level forward kinematics for Kinova Gen3

Functions
---------
forward_kinematics(joint_position)

"""

import math
import numpy as np


def forward_kinematics(joint_position):
    """
    Position level forward kinematics of the Kinova Gen3 robot

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot

    Returns
    -------
    ndarray: The end-effector position
    ndarray: The rotation matrix of the end-effector

    """

    x_0 = math.sin(joint_position[0])
    x_1 = math.cos(joint_position[2])
    x_2 = x_0 * x_1
    x_3 = math.cos(joint_position[0])
    x_4 = math.sin(joint_position[1])
    x_5 = x_3 * x_4
    x_6 = math.cos(joint_position[3])
    x_7 = x_5 * x_6
    x_8 = math.cos(joint_position[1])
    x_9 = math.sin(joint_position[2])
    x_10 = x_3 * x_9
    x_11 = x_10 * x_8
    x_12 = math.sin(joint_position[3])
    x_13 = x_0 * x_9
    x_14 = x_1 * x_3
    x_15 = -x_13 + x_14 * x_8
    x_16 = x_12 * x_15
    x_17 = math.cos(joint_position[5])
    x_18 = -x_16 - x_7
    x_19 = x_17 * x_18
    x_20 = math.sin(joint_position[5])
    x_21 = math.sin(joint_position[4])
    x_22 = x_11 + x_2
    x_23 = math.cos(joint_position[4])
    x_24 = -x_12 * x_5 + x_15 * x_6
    x_25 = -x_21 * x_22 + x_23 * x_24
    x_26 = x_20 * x_25
    x_27 = x_0 * x_4
    x_28 = x_27 * x_6
    x_29 = x_13 * x_8
    x_30 = -x_10 - x_2 * x_8
    x_31 = x_12 * x_30
    x_32 = x_28 - x_31
    x_33 = x_17 * x_32
    x_34 = x_14 - x_29
    x_35 = x_12 * x_27 + x_30 * x_6
    x_36 = -x_21 * x_34 + x_23 * x_35
    x_37 = x_20 * x_36
    x_38 = x_4 * x_9
    x_39 = x_6 * x_8
    x_40 = x_1 * x_4
    x_41 = x_12 * x_40
    x_42 = -x_39 + x_41
    x_43 = x_17 * x_42
    x_44 = -x_12 * x_8 - x_40 * x_6
    x_45 = x_21 * x_38 + x_23 * x_44
    x_46 = x_20 * x_45
    x_47 = math.sin(joint_position[6])
    x_48 = x_21 * x_24 + x_22 * x_23
    x_49 = math.cos(joint_position[6])
    x_50 = x_17 * x_25 + x_18 * x_20
    x_51 = x_21 * x_35 + x_23 * x_34
    x_52 = x_17 * x_36 + x_20 * x_32
    x_53 = x_21 * x_44 - x_23 * x_38
    x_54 = x_17 * x_45 + x_20 * x_42

    return np.array(
        [
            -0.0118 * x_0
            - 0.0128 * x_11
            + 0.3143 * x_16
            - 0.1674 * x_19
            - 0.0128 * x_2
            + 0.1674 * x_26
            + 0.4208 * x_5
            + 0.3143 * x_7,
            -0.0128 * x_14
            - 0.4208 * x_27
            - 0.3143 * x_28
            + 0.0128 * x_29
            - 0.0118 * x_3
            + 0.3143 * x_31
            - 0.1674 * x_33
            + 0.1674 * x_37,
            0.0128 * x_38
            + 0.3143 * x_39
            - 0.3143 * x_41
            - 0.1674 * x_43
            + 0.1674 * x_46
            + 0.4208 * x_8
            + 0.2848,
        ]
    ), np.array(
        [
            [-x_47 * x_48 + x_49 * x_50, x_47 * x_50 + x_48 * x_49, -x_19 + x_26],
            [-x_47 * x_51 + x_49 * x_52, x_47 * x_52 + x_49 * x_51, -x_33 + x_37],
            [-x_47 * x_53 + x_49 * x_54, x_47 * x_54 + x_49 * x_53, -x_43 + x_46],
        ]
    )
