# based on SSD
import numpy as np
import tensorflow as tf
import cv2
import random
import os

class Network:
    def __init__(self, image, im_size, is_train):
        self.image = image
        self.im_size = im_size
        self.is_train = is_train

        w1 = tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1)) # /2 - 64
        w2 = tf.Variable(tf.truncated_normal([5,5,16,32], stddev=0.1)) # /4 - 32
        w3 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1)) # /8 - 16 - bb 24
        w4 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1)) # /16 - 8 - bb 48
        w5 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1)) # /32 - 4 - bb 96

        bb_w1 = tf.Variable(tf.truncated_normal([3,3,64, 3*(2 + 4)], stddev=0.1))
        bb_b1 = tf.Variable(tf.constant(0.0, shape=[3*(2 + 4)]))
        bb_w2 = tf.Variable(tf.truncated_normal([3,3,64, 3*(2 + 4)], stddev=0.1))
        bb_b2 = tf.Variable(tf.constant(0.0, shape=[3*(2 + 4)]))
        bb_w3 = tf.Variable(tf.truncated_normal([3,3,64, 3*(2 + 4)], stddev=0.1))
        bb_b3 = tf.Variable(tf.constant(0.0, shape=[3*(2 + 4)]))

        h1 = self.conv(self.image, w1, 2, ['bn', 'relu'])
        h2 = self.conv(h1, w2, 2, ['bn', 'relu'])
        h3 = self.conv(h2, w3, 2, ['bn', 'relu'])
        h4 = self.conv(h3, w4, 2, ['bn', 'relu'])
        h5 = self.conv(h4, w5, 2, ['bn', 'relu'])

        h_bb1 = self.conv(h3, bb_w1, 1, ['None'], 'VALID') + bb_b1
        h_bb2 = self.conv(h4, bb_w2, 1, ['None'], 'VALID') + bb_b2
        h_bb3 = self.conv(h5, bb_w3, 1, ['None'], 'VALID') + bb_b3

        self.output_scale8 = tf.reshape(h_bb1, [-1, 14*14, 3, 2+4])
        self.output_scale16 = tf.reshape(h_bb2, [-1, 6*6, 3, 2+4])
        self.output_scale32 = tf.reshape(h_bb3, [-1, 2*2, 3, 2+4])

    def conv(self, input_features, weight, stride, options, padding='SAME'):
        h = tf.nn.conv2d(input_features, weight, strides=[1, stride, stride, 1], padding=padding)
        if "bn" in options:
            h = tf.layers.batch_normalization(h, training=self.is_train)
        if "relu" in options:
            h = tf.nn.relu(h)
        print h
        return h

class MNIST_OD_Trainer:
    def __init__(self, sess):
        self.sess = sess

        self.im_folder = '/home/nolan/Documents/Data/MNIST_train/8'
        self.files = [os.path.join(r, name) for r,d,f in os.walk(self.im_folder) for name in f if name.endswith(".png")]
        print(self.files[0])
        print("Number of files: %d" % len(self.files))

        self.batch_size = 1

        self.sub_im_size = [28, 28]
        self.im_size = [128, 128]
        self.show_size = (512, 512)

        self.bb_size = {8:14, 16:6, 32:2}
        self.bb_shapes = np.array([[.75, .75], [1., 0.5], [0.5, 1.]])

        self.sub_scale_options = np.arange(0.5, 4, 0.1)
        self.sub_count = range(5)

    def get_bbox(self, im):
        im_binary = np.ceil(im)
        max_r = np.max(im_binary, axis=0)
        max_c = np.max(im_binary, axis=1)

        t_x = np.argmax(max_r)
        t_y = np.argmax(max_c)
        b_x = self.sub_im_size[1] - np.argmax(max_r[::-1])
        b_y = self.sub_im_size[0] - np.argmax(max_c[::-1])

        return [(t_x, t_y), (b_x, b_y)]

    def intersection(self, bbA, bbB):
        bb1 = self.bb_center2corner(bbA)
        bb2 = self.bb_center2corner(bbB)

        xA = max(bb1[0][0], bb2[0][0])
        yA = max(bb1[0][1], bb2[0][1])
        xB = min(bb1[1][0], bb2[1][0])
        yB = min(bb1[1][1], bb2[1][1])

        A1 = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
        A2 = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])

        intersection = (xB - xA + 1e-8) * (yB - yA + 1e-8) / min(A1, A2)
     
        return intersection

    def bb_corner2center(self, bb):
        c_x = np.mean([bb[0][0], bb[1][0]])
        c_y = np.mean([bb[0][1], bb[1][1]])
        w = bb[1][0] - bb[0][0]
        h = bb[1][1] - bb[0][1]
        return [c_x, c_y, w, h]

    def bb_center2corner(self, bb):
        t_x = np.clip(bb[0] - bb[2]/2, 0, 1)
        t_y = np.clip(bb[1] - bb[3]/2, 0, 1)
        b_x = np.clip(bb[0] + bb[2]/2, 0, 1)
        b_y = np.clip(bb[1] + bb[3]/2, 0, 1)

        return [(t_x, t_y), (b_x, b_y)]

    def add_sub_image(self, im, curr_bbs):
        sub_im, sub_bb = self.load_sub_image()

        found_position = False
        for i in range(20):
            scale = random.choice(self.sub_scale_options)
            new_size = (int(self.sub_im_size[0]*scale), int(self.sub_im_size[1]*scale))
            sub_location_options = range(self.im_size[0] - new_size[0])

            sub_r = random.choice(sub_location_options)
            sub_c = random.choice(sub_location_options)

            new_bb = [((sub_bb[0][0]*scale + sub_c)/self.im_size[0], (sub_bb[0][1]*scale + sub_r)/self.im_size[1]),
                      ((sub_bb[1][0]*scale + sub_c)/self.im_size[0], (sub_bb[1][1]*scale + sub_r)/self.im_size[1])]

            new_bb = self.bb_corner2center(new_bb)

            if len(curr_bbs) < 1:
                found_position = True
                break

            for bb in curr_bbs:
                if self.intersection(new_bb, bb) < 0.2:
                    found_position = True
                else:
                    found_position = False
                    break

            if found_position:
                break

        if found_position:
            sub_im = cv2.resize(sub_im, new_size)
            im[sub_r:sub_r+new_size[0], sub_c:sub_c+new_size[1], 0] += sub_im
            curr_bbs.append(new_bb)

        return im, curr_bbs

    def make_image(self):
        im = np.zeros(self.im_size + [1], np.float32)
        bbs = []
        
        for i in range(random.choice(self.sub_count)):
            im, bbs = self.add_sub_image(im, bbs)

        im += np.random.normal(0, 0.2, self.im_size + [1])
        im = np.clip(im, 0, 1)

        return im, bbs

    def load_sub_image(self):
        im = cv2.imread(random.choice(self.files))
        im = im[:,:,0].astype(np.float32) / 255

        bb = self.get_bbox(im)

        return im, bb

    def get_batch(self):
        im_batch = []
        bb_batch = []
        for i in range(self.batch_size):
            im, bbs = self.make_image()

            im_batch.append(im)
            bb_batch.append(bbs)

        return np.asarray(im_batch), np.asarray(bb_batch)

    def draw_bb(self, im, bbs, conf):
        new_im = np.copy(im)
        for i in range(len(bbs)):
            bb = self.bb_center2corner(bbs[i])
            cv2.rectangle(new_im, (int(bb[0][0] * im.shape[0]), int(bb[0][1] * im.shape[1])), 
                                  (int(bb[1][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                                  (0,1,0), 2)
            cv2.putText(new_im,'{0:.2f}'.format(conf[i]), 
                                  (int(bb[0][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,1), 3, cv2.LINE_AA)
        return new_im

    def draw_net_bb(self, im, bb_layer, scale):
        new_im = np.copy(im)
        print bb_layer.shape
        for p in range(bb_layer.shape[0]):
            for b in range(bb_layer.shape[1]):
                conf = bb_layer[p, b, 1]

                if p == 2: # conf > 0.5:
                    p_xy = (np.asarray([p % self.bb_size[scale]+1, p / self.bb_size[scale]+1], dtype=np.float32) + 0.5) * scale / self.im_size[0]
                    p_wh = self.bb_shapes[b] * 3. * scale / self.im_size[0]
                    default_bb = np.empty(4)
                    default_bb[:2] = p_xy
                    default_bb[2:] = p_wh
                    bb_c = default_bb + bb_layer[p, b, 2:6]
                    bb = self.bb_center2corner(bb_c)

                    cv2.rectangle(new_im, (int(bb[0][0] * im.shape[0]), int(bb[0][1] * im.shape[1])), 
                                  (int(bb[1][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                                  (1,0,0), 2)
                    cv2.putText(new_im,'{0:.2f}'.format(conf), 
                                  (int(bb[0][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,1), 3, cv2.LINE_AA)
        return new_im

    def show_images(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.show_size[0]*2, self.show_size[1])

        for i in range(100):

            ims, bbs = self.get_batch()

            image = cv2.cvtColor(cv2.resize(ims[0], self.show_size), cv2.COLOR_GRAY2RGB)
            conf_gt = np.ones(len(bbs[0]), np.float32)
            image_gt = self.draw_bb(image, bbs[0], conf_gt)
            show_im = np.concatenate([image, image_gt], axis=1)
            cv2.imshow("Image", show_im)
            cv2.waitKey(1000)

    def tf_IoU(self, bb1, bb2):
        pass

    def loss(self):
    """
        for each bb - get IoU to all gt_bb
        IoU > 0.5 we call positive
            backprop classification + bb regression
        For all other, pick top k where conf loss is high
            backprop classificaiton
    """
    # find the bbox with best IoU and show it
    # start with just 32



    for i in range(self.output_scale32)

    def train(self):

        self.in_image = tf.placeholder(tf.float32, name = 'in_image',
                        shape= [self.batch_size] + self.im_size + [1])
        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        self.net = Network(self.in_image, self.im_size, self.is_train)
        self.output_scale8 = self.net.output_scale8
        self.output_scale16 = self.net.output_scale16
        self.output_scale32 = self.net.output_scale32

        self.sess.run(tf.global_variables_initializer())

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.show_size[0]*3, self.show_size[1])

        for i in range(100):
            ims, bbs = self.get_batch()

            os8, os16, os32 = self.sess.run([self.output_scale8, self.output_scale16, self.output_scale32],
                                             feed_dict={self.in_image:ims, self.is_train:True})

            image = cv2.cvtColor(cv2.resize(ims[0], self.show_size), cv2.COLOR_GRAY2RGB)
            conf_gt = np.ones(len(bbs[0]), np.float32)
            image_gt = self.draw_bb(image, bbs[0], conf_gt)
            image_net = self.draw_net_bb(image, os32[0], 32)
            show_im = np.concatenate([image, image_gt, image_net], axis=1)
            cv2.imshow("Image", show_im)
            cv2.waitKey(1000)



if __name__ == "__main__":
    with tf.Session() as sess:  
        trainer = MNIST_OD_Trainer(sess)

        # trainer.show_images()
        trainer.train()