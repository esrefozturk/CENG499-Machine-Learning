""" This is an example driver script to test your implementation """

#
from test import *

#
np.random.seed(499)
#

test_affine_forward()
test_affine_backward()
test_relu_forward()
test_relu_backward()
test_L2_loss()
test_ANN_predict()
test_ANN_train_validate()
