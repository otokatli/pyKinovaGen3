'''Test the gravity term of Kinova Gen3

Classes
-------
TestGravity

Functions
---------
test_gravity()

'''

import math
import numpy as np
import numpy.testing as npt
import unittest
from kinova_gen3.gravity import gravity


class TestGravity(unittest.TestCase):
    '''Unit test class for testing the gravity term of Kinova Gen3

    Methods
    -------
    test_gravity()
        Test the gravity term

    '''

    def test_gravity(self):
        '''Test the graivty term of Kinova Gen3'''

        # Test case 1: all joint positions are zero
        joint_pos = np.zeros(7)

        npt.assert_array_almost_equal(gravity(joint_pos),
                                      np.array([0., -0.00025828, 0., -0.00015588, 0., -0.00013982, 0.]))
