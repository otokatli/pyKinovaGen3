import numpy as np
from numpy.linalg import det
from .jacobian import jacobian
from scipy.optimize import approx_fprime


def manipulability(q):
    return np.log(det(jacobian(q) @ jacobian(q).transpose()))


def manipulability_gradient(q):
    return approx_fprime(q, manipulability, np.sqrt(np.finfo(float).eps))
