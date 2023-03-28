"""Jacobian matrix for Kinova Gen3

Functions
---------
jocobian(joint_position)

"""

import math
import numpy as np


def jacobian(joint_position):
    """The Jacobian of the Kinova Gen3 robot

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot

    Returns
    -------
    ndarray: The kinematic Jacobian matrix expressed in the base frame

    """

    x_0 = math.cos(joint_position[0])
    x_1 = math.sin(joint_position[0])
    x_2 = math.sin(joint_position[1])
    x_3 = 0.4208 * x_2
    x_4 = math.cos(joint_position[2])
    x_5 = x_0 * x_4
    x_6 = 0.0128 * x_5
    x_7 = math.cos(joint_position[3])
    x_8 = x_2 * x_7
    x_9 = x_1 * x_8
    x_10 = math.sin(joint_position[2])
    x_11 = math.cos(joint_position[1])
    x_12 = math.sin(joint_position[3])
    x_13 = x_0 * x_10
    x_14 = x_1 * x_4
    x_15 = x_11 * x_14
    x_16 = x_13 + x_15
    x_17 = -x_16
    x_18 = x_1 * x_2 * x_7 - x_12 * x_17
    x_19 = math.cos(joint_position[5])
    x_20 = 0.1674 * x_19
    x_21 = math.sin(joint_position[5])
    x_22 = math.sin(joint_position[4])
    x_23 = x_1 * x_10
    x_24 = -x_0 * x_4 + x_11 * x_23
    x_25 = -x_24
    x_26 = -x_22 * x_25
    x_27 = math.cos(joint_position[4])
    x_28 = x_12 * x_2
    x_29 = x_1 * x_28
    x_30 = x_17 * x_7
    x_31 = x_29 + x_30
    x_32 = x_26 + x_27 * x_31
    x_33 = 0.4208 * x_11
    x_34 = 0.0128 * x_13
    x_35 = x_11 * x_7
    x_36 = x_0 * x_35
    x_37 = x_28 * x_5
    x_38 = x_2 * x_22
    x_39 = x_11 * x_12
    x_40 = 0.1674 * x_21
    x_41 = 0.0128 * x_23
    x_42 = x_11 * x_13
    x_43 = x_14 + x_42
    x_44 = -x_43
    x_45 = 0.3143 * x_12
    x_46 = x_12 * x_20
    x_47 = -x_0 * x_11 * x_4 + x_23
    x_48 = -x_47
    x_49 = x_0 * x_28
    x_50 = x_48 * x_7
    x_51 = -x_0 * x_12 * x_2 + x_50
    x_52 = x_0 * x_8
    x_53 = -x_12 * x_48 - x_52
    x_54 = x_40 * x_53
    x_55 = x_22 * x_51 + x_27 * x_43
    x_56 = -x_22 * x_43 + x_27 * x_51
    x_57 = x_12 * x_47
    x_58 = x_18 * x_40
    x_59 = x_25 * x_27
    x_60 = 0.3143 * x_8
    x_61 = x_39 * x_4
    x_62 = x_2 * x_4
    x_63 = x_10 * x_28
    x_64 = x_39 + x_4 * x_8
    x_65 = x_28 * x_4 - x_35
    x_66 = x_10 * x_2
    x_67 = x_27 * x_66
    x_68 = -x_64
    x_69 = x_12 * x_16 + x_9
    x_70 = -x_16 * x_7 + x_29

    return np.array(
        [
            [
                -0.0118 * x_0
                + 0.0128 * x_1 * x_10 * x_11
                - x_1 * x_3
                + 0.3143 * x_12 * x_17
                - x_18 * x_20
                + 0.1674 * x_21 * x_32
                - x_6
                - 0.3143 * x_9,
                x_0 * x_33
                + x_2 * x_34
                - x_20 * (-x_36 + x_37)
                + 0.3143 * x_36
                - 0.3143 * x_37
                + x_40 * (x_13 * x_38 + x_27 * (-x_0 * x_39 - x_5 * x_8)),
                -x_11 * x_6
                + x_40 * (-x_22 * x_48 + x_27 * x_44 * x_7)
                + x_41
                + x_44 * x_45
                + x_44 * x_46,
                x_20 * x_51 + x_27 * x_54 - 0.3143 * x_49 + 0.3143 * x_50,
                -x_40 * x_55,
                x_20 * x_56 + x_54,
                0,
            ],
            [
                -x_0 * x_3
                + 0.0118 * x_1
                + 0.0128 * x_14
                - x_20 * (x_52 - x_57)
                + x_40 * (-x_22 * x_44 + x_27 * (x_47 * x_7 + x_49))
                + 0.0128 * x_42
                - 0.3143 * x_52
                + 0.3143 * x_57,
                0.3143 * x_1 * x_12 * x_2 * x_4
                - x_1 * x_33
                - 0.3143 * x_1 * x_35
                - x_2 * x_41
                - x_20 * (x_1 * x_11 * x_7 - x_14 * x_28)
                + 0.1674 * x_21 * (-x_23 * x_38 + x_27 * (x_1 * x_39 + x_14 * x_8)),
                0.0128 * x_15
                + x_24 * x_45
                + x_24 * x_46
                + x_34
                + x_40 * (-x_17 * x_22 + x_24 * x_27 * x_7),
                x_20 * x_31 + x_27 * x_58 + 0.3143 * x_29 + 0.3143 * x_30,
                x_40 * (-x_22 * x_31 - x_59),
                x_20 * x_32 + x_58,
                0,
            ],
            [
                0,
                0.0128 * x_10 * x_11
                - x_20 * (x_61 + x_8)
                + 0.1674 * x_21 * (x_10 * x_11 * x_22 + x_27 * (x_28 - x_35 * x_4))
                - x_3
                - x_60
                - 0.3143 * x_61,
                x_20 * x_63
                + x_40 * (x_10 * x_27 * x_8 + x_22 * x_62)
                + 0.0128 * x_62
                + 0.3143 * x_63,
                -x_20 * x_64 + 0.1674 * x_21 * x_27 * x_65 - 0.3143 * x_39 - x_4 * x_60,
                x_40 * (-x_22 * x_68 + x_67),
                x_20 * (x_22 * x_66 + x_27 * x_68) + x_40 * x_65,
                0,
            ],
            [0, x_1, -x_0 * x_2, x_43, x_53, x_55, x_19 * x_53 - x_21 * x_56],
            [
                0,
                x_0,
                x_1 * x_2,
                x_25,
                x_69,
                x_22 * x_70 + x_59,
                x_19 * x_69 - x_21 * (x_26 + x_27 * x_70),
            ],
            [
                -1,
                0,
                -x_11,
                -x_66,
                x_65,
                -x_22 * x_64 - x_67,
                x_19 * x_65 - x_21 * (x_10 * x_2 * x_22 - x_27 * x_64),
            ],
        ]
    )
