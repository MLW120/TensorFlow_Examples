import tensorflow as tf

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()
with tf.Session() as s:
	s.run(init_op)
	print(s.run(state))
	for i in range(10):
		s.run(update)
		print(s.run(state))
