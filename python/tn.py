from __future__ import print_function
import numpy as np
import theano
from theano.tests import unittest_tools as utt

def functor_x_x(x):
	return theano.tensor.maximum(x, x)

def functor_x_2x(x):
	return theano.tensor.maximum(x, x * 2)

def functor_2x_x(x):
	return theano.tensor.maximum(x * 2, x)

def functor_abs(x):
	return abs(x)

x_val = np.arange(20, dtype=theano.config.floatX)
# x_val[0] will be always 0 for x_val or x_val * 2.

print('Verifying gradient of abs(x)')
utt.verify_grad(functor_abs, [x_val], n_tests=10)

print('Verifying gradient of max(x, x)')
utt.verify_grad(functor_x_x, [x_val], n_tests=10)

print('Verifying gradient of max(2x, x)')
utt.verify_grad(functor_2x_x, [x_val], n_tests=10)

print('Verifying gradient of max(x, 2x)')
utt.verify_grad(functor_x_2x, [x_val], n_tests=10)

print('OK')
