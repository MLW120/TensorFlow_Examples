import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = 3 * x_data + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc = 0.0, scale = 0.1))(y_data)

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as s:
	s.run(init)
	#train_data = []
	for step in range(10000):
		evals = s.run([train,a,b])[1:]
		if step % 1000 == 0:
			print(step,evals)
			print(s.run(loss))
			#train_data.append(evals)
	#print(train_data)
