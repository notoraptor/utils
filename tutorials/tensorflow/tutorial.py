# set TF_CPP_MIN_LOG_LEVEL=2 # disable warnings about avx compilation
# (bad) # set TF_CPP_MIN_VLOG_LEVEL=2 # disable warnings about avx compilation
# python -c "import tensorflow as tf; sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))"
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2.5, dtype=tf.float32)
c1 = tf.placeholder(tf.float32)
c2 = tf.placeholder(tf.float32)
d = tf.add(a, b)
e = c1 + d - c2
session = tf.Session()
session.run(e, {c1: 5, c2:-7})

## Select device.
import tensorflow as tf
with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
with tf.device('/cpu:0'):
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True, to see where ops are executed.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

## Loop.
import tensorflow as tf
def tf_print(t):
    return tf.Print(t, [t], "Current:")
i0 = tf.constant(0)
m0 = tf.ones([1, 2])
condition = lambda i, m: i < 10
body = lambda i, m: [tf_print(i+1), tf.concat([m, m], axis=0)]
out = tf.while_loop(
	condition, body, loop_vars=[i0, m0],
	# we use shape_invariants because shape of second parameter may change (it seems None indicate the changing dimension).
	shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
sess = tf.Session()
final_i, final_m = sess.run(out)
print(final_i)
print(final_m.shape)
print(final_m)

## Profiling with module "timeline": https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
import tensorflow as tf
from tensorflow.python.client import timeline

a = tf.random_normal([2000, 5000])
b = tf.random_normal([5000, 1000])
res = tf.matmul(a, b)

with tf.Session() as sess:
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)


## List available device (is it really work?): https://stackoverflow.com/a/38580201
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
# Note that (at least up to TensorFlow 1.4), calling device_lib.list_local_devices() will run some initialization code that, by default, will allocate all of the GPU memory on all of the devices. To avoid this, first create a session with an explicitly small per_process_gpu_fraction, or allow_growth=True, to prevent all of the memory being allocated. See this question for more details.

## Example usage.
import tensorflow as tf
# Set TensorFlow random seed.
tf.set_random_seed(1234)
# Create two scalar variables, x and y, initialized at random.
x = tf.get_variable(name='x', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
y = tf.get_variable(name='y', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
# Create a tensor z whose value represents the expression
#     2(x - 2)^2 + 2(y + 3)^2
z = 2 * (x - 2) ** 2 + 2 * (y + 3) ** 2
# Create optimization.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_op = optimizer.minimize(loss=z, var_list=tf.trainable_variables())
# Optimize.
with tf.Session() as session:
	# Run the global initializer op for x and y.
	session.run(tf.global_variables_initializer())
	for _ in range(20):
		# Run the update ops for x and y.
		session.run(update_op)
		# Retrieve the values for x, y, and z, and print them.
		x_val, y_val, z_val = session.run([x, y, z])
		print('x = {:4.2f}, y = {:4.2f}, z = {:4.2f}'.format(x_val, y_val, z_val))

# Variable scopes operate on all tensors
with tf.variable_scope('foo'):
    # Scopes can be nested
    with tf.variable_scope('bar'):
        print(tf.get_variable('a', shape=[]).name)
        print(tf.constant(0.0, name='b').name)
# Name scopes do not operate on variables
with tf.name_scope('machine'):
    with tf.name_scope('learning'):
        print(tf.get_variable('a', shape=[]).name)
        print(tf.constant(0.0, name='b').name)
