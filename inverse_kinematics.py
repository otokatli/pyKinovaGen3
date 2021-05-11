import numpy

def inverse_kinematics(q, xp):
    '''
    Velocity level inverse kinematics of the Kinova Gen3 robot
    
    Arguments
    ---------
    q : array_like
        The joint angles of the robot
    xp : array_like
         The end-effector velocities of the robot
        
    Returns
    -------
    ndarray
        The joint velocities
    '''
    
    return numpy.linalg.pinv(jacobian(q)) @ xp
