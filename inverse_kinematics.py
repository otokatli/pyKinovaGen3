import numpy
from .jacobian import jacobian
from .manipulability import manipulability_gradient
from .joint_limits import joint_limits_gradient


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
    

def _nullspace_projector(q):
    return numpy.eye(7) - numpy.linalg.pinv(jacobian(q)) @ jacobian(q)


def multicriteria_ik(q, xp):
    z_manipulability = manipulability_gradient(q)
    z_joint_limits = joint_limits_gradient(q)
    
    alpha = 0.75
    k_z = 5
    
    z = alpha * z_manipulability + (1 - alpha) * z_joint_limits
    
    return inverse_kinematics(q, xp) + k_z * _nullspace_projector(q) @ z


def inverse_kinematics_dls(q, xp, k=0.01):
    """
    Damped least squares solution for inverse kinematics
    """
    J = jacobian(q)
    return J.transpose() @ numpy.linalg.inv(J @ J.transpose() + k**2 * numpy.eye(6)) @ xp


def multicriteria_ik_damped(q, xp):
    z_manipulability = manipulability_gradient(q)
    z_joint_limits = joint_limits_gradient(q)
    
    alpha = 0.75
    k_z = 5
    
    z = alpha * z_manipulability + (1 - alpha) * z_joint_limits
    
    return inverse_kinematics_dls(q, xp) + k_z * _nullspace_projector(q) @ z
