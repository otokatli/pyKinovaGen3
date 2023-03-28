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
    x_14 = x_1 * x_3 * x_8 - x_13
    x_15 = x_12 * x_14
    x_16 = math.cos(joint_position[5])
    x_17 = -x_15 - x_7
    x_18 = x_16 * x_17
    x_19 = math.sin(joint_position[5])
    x_20 = math.sin(joint_position[4])
    x_21 = x_11 + x_2
    x_22 = math.cos(joint_position[4])
    x_23 = -x_12 * x_5 + x_14 * x_6
    x_24 = -x_20 * x_21 + x_22 * x_23
    x_25 = x_19 * x_24
    x_26 = x_0 * x_4
    x_27 = -x_10 - x_2 * x_8
    x_28 = x_0 * x_4 * x_6 - x_12 * x_27
    x_29 = x_16 * x_28
    x_30 = x_1 * x_3 - x_13 * x_8
    x_31 = x_12 * x_26 + x_27 * x_6
    x_32 = -x_20 * x_30 + x_22 * x_31
    x_33 = x_4 * x_9
    x_34 = x_6 * x_8
    x_35 = x_1 * x_4
    x_36 = x_12 * x_35
    x_37 = -x_34 + x_36
    x_38 = x_16 * x_37
    x_39 = -x_12 * x_8 - x_35 * x_6
    x_40 = x_20 * x_33 + x_22 * x_39
    x_41 = x_19 * x_40
    x_42 = math.sin(joint_position[6])
    x_43 = x_20 * x_23 + x_21 * x_22
    x_44 = math.cos(joint_position[6])
    x_45 = x_16 * x_24 + x_17 * x_19
    x_46 = x_20 * x_31 + x_22 * x_30
    x_47 = x_16 * x_32 + x_19 * x_28
    x_48 = x_20 * x_39 - x_22 * x_33
    x_49 = x_16 * x_40 + x_19 * x_37

    return (
        np.array(
            [
                [
                    -0.0118 * x_0
                    - 0.0128 * x_11
                    + 0.3143 * x_15
                    - 0.1674 * x_18
                    - 0.0128 * x_2
                    + 0.1674 * x_25
                    + 0.4208 * x_5
                    + 0.3143 * x_7
                ],
                [
                    0.0128 * x_0 * x_8 * x_9
                    - 0.0128 * x_1 * x_3
                    + 0.3143 * x_12 * x_27
                    + 0.1674 * x_19 * x_32
                    - 0.3143 * x_26 * x_6
                    - 0.4208 * x_26
                    - 0.1674 * x_29
                    - 0.0118 * x_3
                ],
                [
                    0.0128 * x_33
                    + 0.3143 * x_34
                    - 0.3143 * x_36
                    - 0.1674 * x_38
                    + 0.1674 * x_41
                    + 0.4208 * x_8
                    + 0.2848
                ],
            ]
        ),
        np.array(
            [
                [-x_42 * x_43 + x_44 * x_45, x_42 * x_45 + x_43 * x_44, -x_18 + x_25],
                [
                    -x_42 * x_46 + x_44 * x_47,
                    x_42 * x_47 + x_44 * x_46,
                    x_19 * x_32 - x_29,
                ],
                [-x_42 * x_48 + x_44 * x_49, x_42 * x_49 + x_44 * x_48, -x_38 + x_41],
            ]
        ),
    )
