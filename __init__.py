from .forward_kinematics import forward_kinematics
from .inverse_kinematics import inverse_kinematics
from .jacobian import jacobian
from .gravity import gravity
from .mass_matrix import mass_matrix
from .coriolis import coriolis


class KinovaGen3:
    def __init__(self):
        pass
    
    def forward_kinematics(self, q):
        return forward_kinematics(q)
    
    def inverse_kinematics(self, q):
        return inverse_kinematics(q)
    
    def jacobian(self, q):
        return jacobian(q)
    
    def mass_matrix(self, q):
        return mass_matrix(q)
    
    def coriolis(self, q, qp):
        return coriolis(q, qp)
    
    def gravity(self, q):
        return gravity(q)
    