"""Jacobian matrix for Kinova Gen3

Functions
---------
jacobian(joint_position)

"""

import math
import numpy


def jacobian(q):
    """The Jacobian of the Kinova Gen3 robot

    Arguments
    ---------
    joint_position (array_like): The joint angles of the robot

    Returns
    -------
    ndarray: The kinematic Jacobian matrix expressed in the base frame

    """

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    # q7 = q[6]

    x0 = math.cos(q1)
    x1 = math.sin(q1)
    x2 = math.sin(q2)
    x3 = 0.42076 * x2
    x4 = math.cos(q3)
    x5 = x0 * x4
    x6 = 0.01275 * x5
    x7 = math.cos(q4)
    x8 = x2 * x7
    x9 = x1 * x8
    x10 = math.sin(q3)
    x11 = math.cos(q2)
    x12 = math.cos(q5)
    x13 = x1 * x10
    x14 = -x0 * x4 + x11 * x13
    x15 = -x14
    x16 = x12 * x15
    x17 = math.sin(q4)
    x18 = x0 * x10
    x19 = x1 * x4
    x20 = x11 * x19
    x21 = x18 + x20
    x22 = -x21
    x23 = math.sin(q5)
    x24 = x17 * x2
    x25 = x1 * x24
    x26 = x22 * x7
    x27 = x25 + x26
    x28 = x23 * x27
    x29 = x1 * x2 * x7 - x17 * x22
    x30 = math.cos(q6)
    x31 = 0.167455 * x30
    x32 = math.sin(q6)
    x33 = x15 * x23
    x34 = -x33
    x35 = x12 * x27
    x36 = x34 + x35
    x37 = 0.42076 * x11
    x38 = 0.01275 * x18
    x39 = x11 * x7
    x40 = x0 * x39
    x41 = 0.0003501 * x12
    x42 = x2 * x41
    x43 = x24 * x5
    x44 = x11 * x17
    x45 = -x0 * x44 - x5 * x8
    x46 = 0.0003501 * x23
    x47 = x2 * x23
    x48 = 0.167455 * x32
    x49 = 0.01275 * x13
    x50 = -x0 * x11 * x4 + x13
    x51 = -x50
    x52 = x11 * x18
    x53 = x19 + x52
    x54 = -x53
    x55 = 0.31436 * x17
    x56 = x17 * x31
    x57 = x23 * x54
    x58 = x0 * x24
    x59 = -x0 * x17 * x2 + x51 * x7
    x60 = x0 * x8
    x61 = -x17 * x51 - x60
    x62 = x23 * x53
    x63 = x12 * x59
    x64 = x12 * x53 + x23 * x59
    x65 = -x62 + x63
    x66 = x17 * x50
    x67 = x50 * x7 + x58
    x68 = x1 * x44 + x19 * x8
    x69 = x14 * x7
    x70 = x29 * x48
    x71 = 0.31436 * x8
    x72 = x4 * x44
    x73 = x24 - x39 * x4
    x74 = x2 * x4
    x75 = x10 * x24
    x76 = x10 * x8
    x77 = x4 * x8 + x44
    x78 = x24 * x4 - x39
    x79 = x10 * x2
    x80 = -x77
    x81 = x12 * x80
    x82 = x12 * x79
    x83 = x17 * x21 + x9
    x84 = -x21 * x7 + x25

    analytical_jacobian = numpy.array(
        [
            [
                -0.01175 * x0
                + 0.01275 * x1 * x10 * x11
                - x1 * x3
                - 0.0003501 * x16
                + 0.31436 * x17 * x22
                - 0.0003501 * x28
                - x29 * x31
                + 0.167455 * x32 * x36
                - x6
                - 0.31436 * x9,
                x0 * x37
                + x18 * x42
                + x2 * x38
                - x31 * (-x40 + x43)
                + 0.31436 * x40
                - 0.31436 * x43
                - x45 * x46
                + x48 * (x12 * x45 + x18 * x47),
                -x11 * x6
                - x41 * x51
                + x48 * (x12 * x54 * x7 - x23 * x51)
                + x49
                + x54 * x55
                + x54 * x56
                - 0.0003501 * x57 * x7,
                0.167455 * x12 * x32 * x61
                + x31 * x59
                - x46 * x61
                + 0.31436 * x51 * x7
                - 0.31436 * x58,
                -x48 * x64 + 0.0003501 * x62 - 0.0003501 * x63,
                x31 * x65 + x48 * x61,
                0,
            ],
            [
                -x0 * x3
                + 0.01175 * x1
                - 0.0003501 * x12 * x54
                + 0.01275 * x19
                - x31 * (x60 - x66)
                - x46 * x67
                + x48 * (x12 * x67 - x57)
                + 0.01275 * x52
                - 0.31436 * x60
                + 0.31436 * x66,
                0.31436 * x1 * x17 * x2 * x4
                - x1 * x37
                - 0.31436 * x1 * x39
                - x13 * x42
                - x2 * x49
                - x31 * (x1 * x11 * x7 - x19 * x24)
                + 0.167455 * x32 * (x12 * x68 - x13 * x47)
                - x46 * x68,
                x14 * x55
                + x14 * x56
                + 0.01275 * x20
                - x22 * x41
                + x38
                - x46 * x69
                + x48 * (x12 * x69 - x22 * x23),
                x12 * x70 + 0.31436 * x25 + 0.31436 * x26 + x27 * x31 - x29 * x46,
                0.0003501 * x33 - 0.0003501 * x35 + x48 * (-x16 - x28),
                x31 * x36 + x70,
                0,
            ],
            [
                0,
                0.0003501 * x10 * x11 * x12
                + 0.01275 * x10 * x11
                - x3
                - x31 * (x72 + x8)
                + 0.167455 * x32 * (x10 * x11 * x23 + x12 * x73)
                - x46 * x73
                - x71
                - 0.31436 * x72,
                x31 * x75
                + x41 * x74
                - x46 * x76
                + x48 * (x12 * x76 + x23 * x74)
                + 0.01275 * x74
                + 0.31436 * x75,
                0.167455 * x12 * x32 * x78
                - x31 * x77
                - x4 * x71
                - 0.31436 * x44
                - x46 * x78,
                0.167455 * x32 * (-x23 * x80 + x82) - x46 * x79 - 0.0003501 * x81,
                x31 * (x23 * x79 + x81) + x48 * x78,
                0,
            ],
            [0, x1, -x0 * x2, x53, x61, x64, x30 * x61 - x32 * x65],
            [
                0,
                x0,
                x1 * x2,
                x15,
                x83,
                x16 + x23 * x84,
                x30 * x83 - x32 * (x12 * x84 + x34),
            ],
            [
                -1,
                0,
                -x11,
                -x79,
                x78,
                -x23 * x77 - x82,
                x30 * x78 - x32 * (x10 * x2 * x23 - x12 * x77),
            ],
        ]
    )

    return analytical_jacobian
