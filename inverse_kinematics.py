import numpy

def inverse_kinematics(q, qp):
    '''
    Velocity level inverse kinematics of the Kinova Gen3 robot
    
    Arguments
    ---------
    q : array_like
        The joint angles of the robot
    qp : array_like
         The joint velocities of the robot
        
    Returns
    -------
    ndarray
        The end-effector velocity
    '''
    
    return numpy.linalg.pinv(jacobian(q)) @ qp
