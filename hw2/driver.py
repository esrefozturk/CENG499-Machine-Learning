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
