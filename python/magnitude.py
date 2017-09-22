from __future__ import absolute_import, print_function, division

import math

def order_of_magnitude(floating_value):
	if floating_value == 0:
		return 0
	log10_val = math.log10(abs(floating_value))
	return -math.ceil(-log10_val) if log10_val < 0 else math.floor(log10_val)

def get_magnitude_scale(val, tol):
	val_magnitude = order_of_magnitude(val)
	tol_magnitude = order_of_magnitude(tol)
	return tol_magnitude - val_magnitude

def scale_to_tolerance(val, tol):
	return val * 10 ** get_magnitude_scale(val, tol)

def test_order_of_magnitude():
	def run_test(c, e):
		computed = order_of_magnitude(c)
		assert computed == e, "Expected order_of_magnitude(%s) == %s, got %s" % (c, e, computed)
	for test_case, expected in (
		(0, 0),
		(1, 0),
		(2, 0),
		(10, 1),
		(12, 1),
		(1000, 3),
		(1024, 3),
		(12.345, 1),
		(0.1, -1),
		(0.5, -1),
		(-0.1, -1),
		(-0.5, -1),
		(1e-5, -5),
		(9.99999e-6, -6),
		(-12.34, 1),
		(-0.0007, -4),
		(-1, 0),
		(-1000, 3),
	):
		yield (run_test, test_case, expected)
