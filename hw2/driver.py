""" This is an example driver script to test your implementation """

import numpy as np
import matplotlib.pyplot as plt

#
from layers import *
from test import *
from ann import *
from data import *
#
np.random.seed(499)
#

# Test the affine_forward function
test_affine_forward()
test_affine_backward()
test_relu_forward()
test_relu_backward()
test_L2_loss()
test_ANN_predict()
