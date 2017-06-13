from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

save_folder = "4exp/"

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph(save_folder + "saved_models/mnist_4experts.tfrecords.meta")

saver.restore(sess, save_folder + "saved_models/mnist_4experts.tfrecords")
print "Restored"

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
print x, "\n", y_, "\n", keep_prob

accuracy = tf.get_collection('accuracy')[0]


print("test accuracy %g"%sess.run(accuracy, feed_dict={
				x:mnist.test.images[:1000], 
				y_:mnist.test.labels[:1000], 
				keep_prob: 1.0}))


