import tensorflow as tf
import numpy as np
import datetime
import sys
# Import for profiling.
from tensorflow.python.client import timeline

# Using console args to set manual XLA activation for matmul operations.
if len(sys.argv) == 2 and sys.argv[1] == 'xla':
	jit_level = tf.OptimizerOptions.ON_1
	print('XLA activated.')
else:
	print('XLA **NOT** activated.')
	jit_level = 0
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

np.random.seed(1234)
a = tf.placeholder(dtype=tf.float32, shape=[11, 15])
b = tf.placeholder(dtype=tf.float32, shape=[15, 44])
c = tf.placeholder(dtype=tf.float32, shape=[44, 64])
if jit_level:
	with jit_scope():
		mul = tf.matmul(tf.matmul(a, b), c)
else:
	mul = tf.matmul(tf.matmul(a, b), c)
with tf.Session() as sess:
	# Options for profiling.
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()

	aval = np.random.random(size=[11, 15])
	bval = np.random.random(size=[15, 44])
	cval = np.random.random(size=[44, 64])

	# Run session with profiling.
	result = sess.run(mul, feed_dict={a: aval, b: bval, c:cval}, options=options, run_metadata=run_metadata)
	# Get profile to a chrome trace format.
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	current_time = datetime.datetime.now().isoformat()
	trace_filename = 'trace-%s.json' % current_time
	print('Profile trace in', trace_filename)
	with open(trace_filename, 'w') as f:
		f.write(chrome_trace)
	
	print(type(result).__name__, result.shape)
