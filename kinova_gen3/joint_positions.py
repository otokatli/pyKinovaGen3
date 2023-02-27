from numpy import array, cos, sin


def joint_positions(q):
    '''
    Positions of the joints of Kinova Gen3 robot in base frame
    
    Arguments
    ---------
    q : array_like
        The joint angles of the robot
        
    Returns
    -------
    ndarray
        The joint positions
    '''
    
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]
    
    x0 = sin(q1)
    x1 = cos(q1)
    x2 = -0.0118*x0
    x3 = sin(q2)
    x4 = 0.2104*x3
    x5 = -0.0118*x1
    x6 = cos(q2)
    x7 = cos(q3)
    x8 = x0*x7
    x9 = sin(q3)
    x10 = x1*x9
    x11 = x10*x6
    x12 = 0.4208*x3
    x13 = x1*x12 + x2
    x14 = x1*x7
    x15 = x0*x9
    x16 = x15*x6
    x17 = -x0*x12 + x5
    x18 = x3*x9
    x19 = 0.4208*x6 + 0.2848
    x20 = sin(q4)
    x21 = x14*x6 - x15
    x22 = x20*x21
    x23 = cos(q4)
    x24 = x1*x3
    x25 = x23*x24
    x26 = -0.0128*x11 + x13 - 0.0128*x8
    x27 = -x10 - x6*x8
    x28 = x20*x27
    x29 = x0*x3
    x30 = x23*x29
    x31 = -0.0128*x14 + 0.0128*x16 + x17
    x32 = x23*x6
    x33 = x3*x7
    x34 = x20*x33
    x35 = 0.0128*x18 + x19
    x36 = 0.3143*x22 + 0.3143*x25 + x26
    x37 = 0.3143*x28 - 0.3143*x30 + x31
    x38 = 0.3143*x32 - 0.3143*x34 + x35
    x39 = sin(q5)
    x40 = cos(q5)
    x41 = 0.1059*sin(q6)
    x42 = 0.1059*cos(q6)
    
    return array([[0, 0, 0.1564],
                  [-0.0054*x0, -0.0054*x1, 0.2848],
                  [x1*x4 + x2, -x0*x4 + x5, 0.2104*x6 + 0.2848],
                  [-0.0064*x11 + x13 - 0.0064*x8, -0.0064*x14 + 0.0064*x16 + x17, 0.0064*x18 + x19],
                  [0.2084*x22 + 0.2084*x25 + x26, 0.2084*x28 - 0.2084*x30 + x31, 0.2084*x32 - 0.2084*x34 + x35],
                  [x36, x37, x38],
                  [x36 + x41*(-x39*(x11 + x8) + x40*(-x20*x24 + x21*x23)) - x42*(-x22 - x25), x37 + x41*(-x39*(x14 - x16) + x40*(x20*x29 + x23*x27)) - x42*(-x28 + x30), x38 + x41*(x18*x39 + x40*(-x20*x6 - x23*x33)) - x42*(-x32 + x34)]])
