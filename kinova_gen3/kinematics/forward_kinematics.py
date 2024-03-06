"""Position level forward kinematics for Kinova Gen3

Functions
---------
forward_kinematics(joint_position)

"""

import math
import numpy


def forward_kinematics(q):
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

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]

    x0 = math.sin(q1)
    x1 = math.cos(q3)
    x2 = x0 * x1
    x3 = math.sin(q2)
    x4 = math.cos(q1)
    x5 = math.cos(q4)
    x6 = math.cos(q2)
    x7 = math.sin(q3)
    x8 = x4 * x7
    x9 = x6 * x8
    x10 = math.cos(q5)
    x11 = x2 + x9
    x12 = x10 * x11
    x13 = math.sin(q4)
    x14 = x0 * x7
    x15 = x1 * x4 * x6 - x14
    x16 = math.sin(q5)
    x17 = x3 * x4
    x18 = -x13 * x17 + x15 * x5
    x19 = x16 * x18
    x20 = math.cos(q6)
    x21 = -x13 * x15 - x17 * x5
    x22 = x20 * x21
    x23 = math.sin(q6)
    x24 = x10 * x18 - x11 * x16
    x25 = x0 * x3
    x26 = x1 * x4 - x14 * x6
    x27 = x10 * x26
    x28 = -x2 * x6 - x8
    x29 = x13 * x25 + x28 * x5
    x30 = x16 * x29
    x31 = x0 * x3 * x5 - x13 * x28
    x32 = x20 * x31
    x33 = x10 * x29 - x16 * x26
    x34 = x3 * x7
    x35 = x5 * x6
    x36 = x10 * x34
    x37 = x1 * x3
    x38 = x13 * x37
    x39 = -x35 + x38
    x40 = x20 * x39
    x41 = -x13 * x6 - x37 * x5
    x42 = x16 * x41
    x43 = x10 * x41 + x16 * x34
    x44 = x23 * x43
    x45 = math.sin(q7)
    x46 = x12 + x19
    x47 = math.cos(q7)
    x48 = x20 * x24 + x21 * x23
    x49 = x27 + x30
    x50 = x20 * x33 + x23 * x31
    x51 = -x36 + x42
    x52 = x20 * x43 + x23 * x39

    position = numpy.array(
        [
            [
                -0.01175 * x0
                - 0.0003501 * x12
                + 0.31436 * x13 * x15
                - 0.0003501 * x19
                - 0.01275 * x2
                - 0.167455 * x22
                + 0.167455 * x23 * x24
                + 0.31436 * x3 * x4 * x5
                + 0.42076 * x3 * x4
                - 0.01275 * x9
            ],
            [
                0.01275 * x0 * x6 * x7
                - 0.01275 * x1 * x4
                + 0.31436 * x13 * x28
                + 0.167455 * x23 * x33
                - 0.31436 * x25 * x5
                - 0.42076 * x25
                - 0.0003501 * x27
                - 0.0003501 * x30
                - 0.167455 * x32
                - 0.01175 * x4
            ],
            [
                0.01275 * x34
                + 0.31436 * x35
                + 0.0003501 * x36
                - 0.31436 * x38
                - 0.167455 * x40
                - 0.0003501 * x42
                + 0.167455 * x44
                + 0.42076 * x6
                + 0.28481
            ],
        ]
    ).flatten()

    rotation = numpy.array(
        [
            [x45 * x46 - x47 * x48, -x45 * x48 - x46 * x47, -x22 + x23 * x24],
            [x45 * x49 - x47 * x50, -x45 * x50 - x47 * x49, x23 * x33 - x32],
            [x45 * x51 - x47 * x52, -x45 * x52 - x47 * x51, -x40 + x44],
        ]
    )

    return position, rotation
