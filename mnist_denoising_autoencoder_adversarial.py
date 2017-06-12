# mnist autoencoder
# https://github.com/JosephCatrambone/ImageFab/blob/master/train_model.py

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

    ####### build the model - auto encoder #############################################
    print "=========================================================================="
    #### Genorator
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 28x28
    print x_image
    noise = tf.truncated_normal(shape=tf.shape(x_image), stddev=0.25)
    x_noise = x_image + noise

    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])

    W_conv2 = weight_variable([5, 5, 16, 24])
    b_conv2 = bias_variable([24])

    W_conv3 = weight_variable([3, 3, 24, 32])
    b_conv3 = bias_variable([32])

    W_dconv1 = weight_variable([3, 3, 24, 32])
    b_dconv1 = bias_variable([24])

    W_dconv2 = weight_variable([5, 5, 16, 24])
    b_dconv2 = bias_variable([16])

    W_dconv3 = weight_variable([5, 5, 1, 16])
    b_dconv3 = bias_variable([1])

    G_Vars = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3,
                W_dconv1, b_dconv1, W_dconv2, b_dconv2, W_dconv3, b_dconv3]

    def generator(image_input):
        h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1) + b_conv1, name="conv1")
        h_pool1 = max_pool_2x2(h_conv1) # 14x14

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="conv2")
        h_pool2 = max_pool_2x2(h_conv2) # 7x7
        
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name="conv3")

        osize_dconv1 = [batch_size, 7, 7, 24]
        h_dconv1 = tf.nn.relu(conv2d_transpose(h_conv3, W_dconv1, osize_dconv1) + b_dconv1, name="dconv1")
        
        h_unpool1 = unpool_2x2(h_dconv1) # 14x14
        osize_dconv2 = [batch_size, 14, 14, 16]
        h_dconv2 = tf.nn.relu(conv2d_transpose(h_unpool1, W_dconv2, osize_dconv2) + b_dconv2, name="dconv2")
        
        h_unpool2 = unpool_2x2(h_dconv2) # 28x28
        osize_dconv3 = [batch_size, 28, 28, 1]
        h_dconv3 = tf.nn.sigmoid(conv2d_transpose(h_unpool2, W_dconv3, osize_dconv3) + b_dconv3, name="dconv3")

        generated_image = h_dconv3
        return generated_image

    ### Descriminator
    W_conv1d = weight_variable([5, 5, 1, 16])
    b_conv1d = bias_variable([16])

    W_conv2d = weight_variable([5, 5, 16, 24])
    b_conv2d = bias_variable([24])

    W_fc1d = weight_variable([7 * 7 * 24, 256])
    b_fc1d = bias_variable([256])

    W_fc2d = weight_variable([256, 1])
    b_fc2d = bias_variable([1])

    D_Vars = [W_conv1d, b_conv1d, W_conv2d, b_conv2d, W_fc1d, b_fc1d, W_fc2d, b_fc2d]

    def discriminator(image_input):
        h_conv1d = tf.nn.relu(conv2d(image_input, W_conv1d) + b_conv1d, name="conv1d")
        h_pool1d = max_pool_2x2(h_conv1d) # 14x14

        h_conv2d = tf.nn.relu(conv2d(h_pool1d, W_conv2d) + b_conv2d, name="conv2d")
        h_pool2d = max_pool_2x2(h_conv2d) # 7x7
        h_pool2d_flat = tf.reshape(h_pool2d, [-1, 7*7*24])

        h_fc1d = tf.nn.relu(tf.matmul(h_pool2d_flat, W_fc1d) + b_fc1d, name="fc1d")

        h_fc2d = (tf.matmul(h_fc1d, W_fc2d) + b_fc2d) # center representation

        how_real = tf.nn.sigmoid(h_fc2d)
        return how_real, h_fc2d


    y_image = generator(x_noise)
    fake_decision, fake_logit = discriminator(y_image)
    real_decision, real_logit = discriminator(x_image)

    show_num_parameters()

    G_l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(x_image - y_image), 3))
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logit, tf.ones_like(real_logit)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.zeros_like(fake_logit)))
    D_loss = D_loss_real + D_loss_fake
    G_Ad_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.ones_like(fake_logit)))
    G_loss = G_l1_loss# + 0.005 * G_Ad_loss


    ########################################################################
    train_step_D = tf.train.AdamOptimizer(1e-6).minimize(D_loss, var_list=D_Vars)
    train_step_G = tf.train.AdamOptimizer(1e-6).minimize(G_loss, var_list=G_Vars)

    save_folder = "daea_saved/"

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
                idloss, igloss, igl1, igad, ix_image,ix_noise, iy_image, ifake_decision, ireal_decision = sess.run(
                    [D_loss, G_loss, G_l1_loss, G_Ad_loss, x_image, x_noise, y_image, fake_decision, real_decision], feed_dict={x:batch[0]})
                print "step",i, "loss", idloss, igloss, igl1, igad, "duration", datetime.datetime.now() - start_time

                print "Real Image", np.max(ix_image[0]), np.min(ix_image[0]), \
                        "CNN Image", np.max(iy_image[0]), np.min(iy_image[0]), \
                        "Fake", ifake_decision[0], "Real", ireal_decision[0]
                show_im = np.concatenate([ix_image[0], ix_noise[0], iy_image[0]], axis=1)
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
