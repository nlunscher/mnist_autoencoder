# http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/

print "Starting Program..."

import cv2
import tensorflow as tf
import numpy as np
import datetime
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def write_to_csv(filename, number):
    with open(filename, 'a') as f:
        f.write(str(number) + ',\n')

def show_num_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        # print(variable_parametes)
        total_parameters += variable_parametes
    print "Total trainable model parameters:", total_parameters
    return total_parameters

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

def conv2d_transpose(x, W, out_shape):
    return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1],
                                    padding='SAME', output_shape=out_shape)

def unpool_2x2(x):
    size = x.get_shape().as_list()
    return tf.image.resize_images(x, [size[1] * 2, size[2] * 2])

with tf.Graph().as_default():

    train_iterations = 600000
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[None, 784])
    real_image = tf.reshape(x, [-1, 28, 28, 1]) # 28x28

    ####### build the model - auto encoder #############################################
    print "=========================================================================="
    input_noise = tf.random_uniform(shape=[batch_size, 128])    

    W_fc1 = weight_variable([128, 7 * 7 * 64])
    b_fc1 = bias_variable([7 * 7 * 64])
    
    W_dconv1 = weight_variable([5, 5, 32, 64])
    b_dconv1 = bias_variable([32])
    osize_dconv1 = [-1, 14, 14, 32]
    osize_dconv1[0] = batch_size

    W_dconv2 = weight_variable([5, 5, 1, 32])
    b_dconv2 = bias_variable([1])
    osize_dconv2 = real_image.get_shape().as_list()
    osize_dconv2[0] = batch_size

    G_Vars = [W_fc1, b_fc1, W_dconv1, b_dconv1, W_dconv2, b_dconv2]

    def generator(encoding_input):
        h_fc1 = tf.nn.relu(tf.matmul(input_noise, W_fc1) + b_fc1, name="fc1")
        print h_fc1

        h_fc1_2d = tf.reshape(h_fc1, [-1, 7, 7, 64]) # 7x7
        print h_fc1_2d
        h_unpool1 = unpool_2x2(h_fc1_2d) # 14x14
        print h_unpool1

        h_dconv1 = tf.nn.relu(conv2d_transpose(h_unpool1, W_dconv1, osize_dconv1) + b_dconv1, name="dconv1")
        print h_dconv1

        h_unpool2 = unpool_2x2(h_dconv1) # 28x28
        print h_unpool2

        h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_unpool2, W_dconv2, osize_dconv2)
                        + b_dconv2, name="dconv2")
        print h_dconv2
        generated_image = h_dconv2

        return generated_image


    ### adversary
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_fc2 = weight_variable([7 * 7 * 64, 1024])
    b_fc2 = bias_variable([1024])

    W_fc3 = weight_variable([1024, 1])
    b_fc3 = bias_variable([1])

    D_Vars = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc2, b_fc2, W_fc3, b_fc3]

    def discriminator(image_input):
        h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1) + b_conv1, name="conv1")
        print h_conv1
        h_pool1 = max_pool_2x2(h_conv1) # 14x14
        print h_pool1

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="conv2")
        print h_conv2
        h_pool2 = max_pool_2x2(h_conv2) # 7x7
        print h_pool2
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2, name="fc2")
        print h_fc2

        h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3, name="fc3") # center representation
        print h_fc3

        how_real = h_fc3
        return how_real


    y_image = generator(input_noise)
    fake_decision = discriminator(y_image)
    real_decision = discriminator(real_image)

    show_num_parameters()

    D_loss = -tf.reduce_mean(tf.log(real_decision) + tf.log(1. - fake_decision))
    G_loss = -tf.reduce_mean(tf.log(fake_decision))
    ########################################################################

    train_step_D = tf.train.AdamOptimizer(1e-6).minimize(D_loss, var_list=D_Vars)
    train_step_G = tf.train.AdamOptimizer(1e-6).minimize(G_loss, var_list=G_Vars)

    save_folder = "gan_saved/"

    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        print "Initiating..."
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 200*3, 200)

        training_start_time = datetime.datetime.now()
        print "Starting, training for", train_iterations, "iterations"
        start_time = datetime.datetime.now()
        for i in range(1, train_iterations + 1):
            batch = mnist.train.next_batch(batch_size)

            if i % 100 == 0 or i == 1:
                idloss, igloss, ir_image, iy_image, ifake_decision, ireal_decision = sess.run(
                        [D_loss, G_loss, real_image, y_image, fake_decision, real_decision], feed_dict={x:batch[0]})
                print "step",i, "loss", idloss, igloss, "duration", datetime.datetime.now() - start_time

                print "Real Image", np.max(ir_image[0]), np.min(ir_image[0]), \
                        "CNN Image", np.max(iy_image[0]), np.min(iy_image[0]), \
                        "Fake", ifake_decision[0], "Real", ireal_decision[0]
                show_im = np.concatenate([ir_image[0], iy_image[0]], axis=1)
                cv2.imshow("Image", show_im)
                key = cv2.waitKey(10)

                if i % 5000 == 0 or i == 1:
                    im_name = save_folder + "saved_images/" + str(i) + "_train_im.tif"
                    show_im[np.where(show_im > 1)] = 1
                    show_im[np.where(show_im < 0.0)] = 0
                    saved = cv2.imwrite(im_name, (show_im*255).astype(np.uint8))
                    print "Saving Images", im_name, saved

            if i % 100000 == 0:
                print "Saving Model"
                save_path = saver.save(sess, save_folder + "saved_models/" + str(i) + "_mnist_auto.tfrecords")
                print ("Saved model as: %s" % save_path)
            # else:
            #     print i

            _,_, idloss, igloss = sess.run([train_step_D, train_step_G, D_loss, G_loss], feed_dict={x: batch[0]})
            to_csv = str(i) + ", " + str(idloss) + ", " + str(igloss)
            write_to_csv('training_losses.csv', to_csv)

        print "Total Duration:", datetime.datetime.now() - start_time

        # print "Testing"
        # print("test accuracy %g"%accuracy.eval(feed_dict={
        #       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        save_path = saver.save(sess, save_folder + "saved_models/mnist_auto_final.tfrecords")
        print ("Saved model as: %s" % save_path)


print "Ending Program..."
