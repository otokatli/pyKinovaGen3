from numpy.linalg import inv
from .jacobian import jacobian


def manipulability(q):
    return inv(jacobian(q) @ jacobian(q).transpose())