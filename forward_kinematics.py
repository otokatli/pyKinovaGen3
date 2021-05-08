from numpy import array, cos, sin


def forward_kinematics(q):
    '''
    Forward kinematics of the Kinova Gen3 robot
    
    Arguments
    ---------
    q : array_like
        The joint angles of the robot
        
    Returns
    -------
    ndarray
        The end-effector position
    ndarray
        The rotation matrix of the end-effector
    '''
    
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]
    
    x0 = sin(q1)
    x1 = cos(q3)
    x2 = x0*x1
    x3 = cos(q1)
    x4 = sin(q2)
    x5 = x3*x4
    x6 = cos(q4)
    x7 = x5*x6
    x8 = cos(q2)
    x9 = sin(q3)
    x10 = x3*x9
    x11 = x10*x8
    x12 = sin(q4)
    x13 = x0*x9
    x14 = x1*x3
    x15 = -x13 + x14*x8
    x16 = x12*x15
    x17 = cos(q6)
    x18 = -x16 - x7
    x19 = x17*x18
    x20 = sin(q6)
    x21 = sin(q5)
    x22 = x11 + x2
    x23 = cos(q5)
    x24 = -x12*x5 + x15*x6
    x25 = -x21*x22 + x23*x24
    x26 = x20*x25
    x27 = x0*x4
    x28 = x27*x6
    x29 = x13*x8
    x30 = -x10 - x2*x8
    x31 = x12*x30
    x32 = x28 - x31
    x33 = x17*x32
    x34 = x14 - x29
    x35 = x12*x27 + x30*x6
    x36 = -x21*x34 + x23*x35
    x37 = x20*x36
    x38 = x4*x9
    x39 = x6*x8
    x40 = x1*x4
    x41 = x12*x40
    x42 = -x39 + x41
    x43 = x17*x42
    x44 = -x12*x8 - x40*x6
    x45 = x21*x38 + x23*x44
    x46 = x20*x45
    x47 = sin(q7)
    x48 = x21*x24 + x22*x23
    x49 = cos(q7)
    x50 = x17*x25 + x18*x20
    x51 = x21*x35 + x23*x34
    x52 = x17*x36 + x20*x32
    x53 = x21*x44 - x23*x38
    x54 = x17*x45 + x20*x42
    return array([-0.0118*x0 - 0.0128*x11 + 0.3143*x16 - 0.1674*x19 - 0.0128*x2 + 0.1674*x26 + 0.4208*x5 + 0.3143*x7, -0.0128*x14 - 0.4208*x27 - 0.3143*x28 + 0.0128*x29 - 0.0118*x3 + 0.3143*x31 - 0.1674*x33 + 0.1674*x37, 0.0128*x38 + 0.3143*x39 - 0.3143*x41 - 0.1674*x43 + 0.1674*x46 + 0.4208*x8 + 0.2848]), array([[-x47*x48 + x49*x50, x47*x50 + x48*x49, -x19 + x26], [-x47*x51 + x49*x52, x47*x52 + x49*x51, -x33 + x37], [-x47*x53 + x49*x54, x47*x54 + x49*x53, -x43 + x46]])
