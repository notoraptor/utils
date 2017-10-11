from __future__ import print_function
import numpy as np
import theano
from theano.tests import unittest_tools as utt

def functor(x):
	z = theano.tensor.maximum(x, x)
	return z

x_val = np.arange(20, dtype=theano.config.floatX).reshape((4, 5)) + 1

print('Verifying ...')
utt.verify_grad(functor, [x_val], n_tests=10)

print('OK')
