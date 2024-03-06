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
    x3 = math.cos(q1)
    x4 = math.sin(q2)
    x5 = x3 * x4
    x6 = math.cos(q4)
    x7 = x5 * x6
    x8 = math.cos(q2)
    x9 = math.sin(q3)
    x10 = x3 * x9
    x11 = x10 * x8
    x12 = math.cos(q5)
    x13 = x11 + x2
    x14 = x12 * x13
    x15 = math.sin(q4)
    x16 = x0 * x9
    x17 = x1 * x3
    x18 = -x16 + x17 * x8
    x19 = x15 * x18
    x20 = math.sin(q5)
    x21 = -x15 * x5 + x18 * x6
    x22 = x20 * x21
    x23 = math.cos(q6)
    x24 = -x19 - x7
    x25 = x23 * x24
    x26 = math.sin(q6)
    x27 = x12 * x21 - x13 * x20
    x28 = x26 * x27
    x29 = x0 * x4
    x30 = x29 * x6
    x31 = x16 * x8
    x32 = x17 - x31
    x33 = x12 * x32
    x34 = -x10 - x2 * x8
    x35 = x15 * x34
    x36 = x15 * x29 + x34 * x6
    x37 = x20 * x36
    x38 = x30 - x35
    x39 = x23 * x38
    x40 = x12 * x36 - x20 * x32
    x41 = x26 * x40
    x42 = x4 * x9
    x43 = x6 * x8
    x44 = x12 * x42
    x45 = x1 * x4
    x46 = x15 * x45
    x47 = -x43 + x46
    x48 = x23 * x47
    x49 = -x15 * x8 - x45 * x6
    x50 = x20 * x49
    x51 = x12 * x49 + x20 * x42
    x52 = x26 * x51
    x53 = math.sin(q7)
    x54 = x14 + x22
    x55 = math.cos(q7)
    x56 = x23 * x27 + x24 * x26
    x57 = x33 + x37
    x58 = x23 * x40 + x26 * x38
    x59 = -x44 + x50
    x60 = x23 * x51 + x26 * x47

    position = numpy.array(
        [
            [
                -0.01175 * x0
                - 0.01275 * x11
                - 0.0003501 * x14
                + 0.31436 * x19
                - 0.01275 * x2
                - 0.0003501 * x22
                - 0.16743 * x25
                + 0.16743 * x28
                + 0.42076 * x5
                + 0.31436 * x7
            ],
            [
                -0.01275 * x17
                - 0.42076 * x29
                - 0.01175 * x3
                - 0.31436 * x30
                + 0.01275 * x31
                - 0.0003501 * x33
                + 0.31436 * x35
                - 0.0003501 * x37
                - 0.16743 * x39
                + 0.16743 * x41
            ],
            [
                0.01275 * x42
                + 0.31436 * x43
                + 0.0003501 * x44
                - 0.31436 * x46
                - 0.16743 * x48
                - 0.0003501 * x50
                + 0.16743 * x52
                + 0.42076 * x8
                + 0.28481
            ],
        ]
    ).flatten()
    
    rotation = numpy.array(
        [
            [-x53 * x54 + x55 * x56, x53 * x56 + x54 * x55, -x25 + x28],
            [-x53 * x57 + x55 * x58, x53 * x58 + x55 * x57, -x39 + x41],
            [-x53 * x59 + x55 * x60, x53 * x60 + x55 * x59, -x48 + x52],
        ]
    )

    return position, rotation
