'''Test the forward kinematics of Kinova Gen3

Classes
-------
TestForwardKinematics

Functions
---------
test_forward_kinematics()

'''

import math
import numpy as np
import numpy.testing as npt
import unittest
from kinova_gen3.forward_kinematics import forward_kinematics


class TestForwardKinematics(unittest.TestCase):
    '''Unit test class for testing the forward kinematics of Kinova Gen3

    Methods
    -------
    test_forward_kinematics()
        Test the position level forward kinematics

    '''

    def test_forward_kinematics(self):
        '''Test the position level forward kinematics of Kinova Gen3'''

        # Test case 1: all joint positions are zero
        joint_pos = np.zeros(7)

        npt.assert_array_equal(forward_kinematics(joint_pos)[0], [0. , -0.0246, 1.1873])
        npt.assert_array_equal(forward_kinematics(joint_pos)[1], np.eye(3))

        # Test case 2: Rotate second joint by pi/2
        joint_pos[1] = math.pi / 2

        npt.assert_allclose(forward_kinematics(joint_pos)[0], [0.9025 , -0.0246, 0.2848])
        npt.assert_allclose(forward_kinematics(joint_pos)[1], \
                np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]), 1e-6, 1e-7)
