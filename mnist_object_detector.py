# based on SSD
import numpy as np
import tensorflow as tf
import cv2
import random
import os
import datetime as dt

class Network:
    def __init__(self, image, im_size, num_bb, is_train):
        self.image = image
        self.im_size = im_size
        self.num_bb = num_bb
        self.is_train = is_train

        self.out_channels = self.num_bb*(2 + 4)

        w1 = tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1)) # /2 - 64
        w2 = tf.Variable(tf.truncated_normal([5,5,16,32], stddev=0.1)) # /4 - 32
        w3 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1)) # /8 - 16 - bb 24
        w4 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1)) # /16 - 8 - bb 48
        w5 = tf.Variable(tf.truncated_normal([5,5,64,64], stddev=0.1)) # /32 - 4 - bb 96

        bb_w1 = tf.Variable(tf.truncated_normal([3,3,64, self.out_channels], stddev=0.1))
        bb_b1 = tf.Variable(tf.constant(0.0, shape=[self.out_channels]))
        bb_w2 = tf.Variable(tf.truncated_normal([3,3,64, self.out_channels], stddev=0.1))
        bb_b2 = tf.Variable(tf.constant(0.0, shape=[self.out_channels]))
        bb_w3 = tf.Variable(tf.truncated_normal([3,3,64, self.out_channels], stddev=0.1))
        bb_b3 = tf.Variable(tf.constant(0.0, shape=[self.out_channels]))

        h1 = self.conv(self.image, w1, 2, ['bn', 'relu'])
        h2 = self.conv(h1, w2, 2, ['bn', 'relu'])
        h3 = self.conv(h2, w3, 2, ['bn', 'relu'])
        h4 = self.conv(h3, w4, 2, ['bn', 'relu'])
        h5 = self.conv(h4, w5, 2, ['bn', 'relu'])

        h_bb1 = self.conv(h3, bb_w1, 1, ['None'], 'VALID') + bb_b1
        h_bb2 = self.conv(h4, bb_w2, 1, ['None'], 'VALID') + bb_b2
        h_bb3 = self.conv(h5, bb_w3, 1, ['None'], 'VALID') + bb_b3

        self.output_scale8 = h_bb1
        self.output_scale16 = h_bb2
        self.output_scale32 = h_bb3

        # softmax the classifications

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
        # print(self.files[0])
        print("Number of files: %d" % len(self.files))

        self.batch_size = 16

        self.sub_im_size = [28, 28]
        self.im_size = [128, 128]
        self.show_size = (512, 512)

        self.bb_size = {8:14, 16:6, 32:2}
        self.bb_shapes = np.array([[.75, .75], 
                                   [1., 0.5], 
                                   [0.5, 1.], 
                                   [1, 1], 
                                   [0.5, 0.5],
                                   [0.75, 0.5],
                                   [0.5, 7.5]])
        self.num_bb = len(self.bb_shapes)
        self.out_channels = self.num_bb*(2 + 4)
        self.default_bb = {}
        for b in self.bb_size:
            self.make_default_bb(b)

        self.sub_scale_options = np.arange(0.9, 4, 0.1)
        self.sub_count = range(1,5)

        self.debug = {}

    def get_bbox(self, im):
        im_binary = np.ceil(im)
        max_r = np.max(im_binary, axis=0)
        max_c = np.max(im_binary, axis=1)

        t_x = np.argmax(max_r)
        t_y = np.argmax(max_c)
        b_x = self.sub_im_size[1] - np.argmax(max_r[::-1])
        b_y = self.sub_im_size[0] - np.argmax(max_c[::-1])

        return [(t_x, t_y), (b_x, b_y)]

    def iou(self, bbA, bbB):
        bb1 = self.bb_center2corner(bbA)
        bb2 = self.bb_center2corner(bbB)

        has_intersection = (    (bb1[0][0] < bb2[1][0] and bb1[0][0] > bb2[0][0] \
                                or bb2[0][0] < bb1[1][0] and bb2[0][0] > bb1[0][0]) \
                            and (bb1[0][1] < bb2[1][1] and bb1[0][1] > bb2[0][1] \
                                or bb2[0][1] < bb1[1][1] and bb2[0][1] > bb1[0][1]))

        xA = max(bb1[0][0], bb2[0][0])
        yA = max(bb1[0][1], bb2[0][1])
        xB = min(bb1[1][0], bb2[1][0])
        yB = min(bb1[1][1], bb2[1][1])

        if has_intersection:
            intersection = (xB - xA) * (yB - yA) #/ min(A1, A2)
        else:
            intersection = 0.0

        A1 = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
        A2 = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])

        IoU = intersection / (A1 + A2 - intersection)

        # if IoU > 1:
        #     print "*************************HIGH IOU"
        #     print bbA, bbB
        #     print bb1, bb2
        #     print intersection, (A1 + A2 - intersection)

        # new_im = np.zeros((500,500, 3))
        # print IoU
        # cv2.rectangle(new_im, (int(bb1[0][0] * new_im.shape[0]), int(bb1[0][1] * new_im.shape[0])), 
        #                       (int(bb1[1][0] * new_im.shape[0]), int(bb1[1][1] * new_im.shape[0])), 
        #                       (0,1,0), 2)
        # cv2.rectangle(new_im, (int(bb2[0][0] * new_im.shape[0]), int(bb2[0][1] * new_im.shape[0])), 
        #                       (int(bb2[1][0] * new_im.shape[0]), int(bb2[1][1] * new_im.shape[0])), 
        #                       (0,0,1), 2)
        # cv2.rectangle(new_im, (int(xA * new_im.shape[0]), int(yA * new_im.shape[0])), 
        #                       (int(xB * new_im.shape[0]), int(yB * new_im.shape[0])), 
        #                       (1,0,0), 2)
        # cv2.imshow("i", new_im)
        # cv2.waitKey(0)
     
        return intersection, IoU

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
                _, iou = self.iou(new_bb, bb)
                if iou < 0.1:
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
        enc_bb_batch = [[], [], []]
        for i in range(self.batch_size):
            im, bbs = self.make_image()

            encoded_bbs8 = self.encode_2d_bb(bbs, 8)
            encoded_bbs16 = self.encode_2d_bb(bbs, 16)
            encoded_bbs32 = self.encode_2d_bb(bbs, 32)

            im_batch.append(im)
            bb_batch.append(bbs)
            enc_bb_batch[0].append(encoded_bbs8)
            enc_bb_batch[1].append(encoded_bbs16)
            enc_bb_batch[2].append(encoded_bbs32)

        enc_bb_batch[0] = np.asarray(enc_bb_batch[0])
        enc_bb_batch[1] = np.asarray(enc_bb_batch[1])
        enc_bb_batch[2] = np.asarray(enc_bb_batch[2])
        return np.asarray(im_batch), np.asarray(bb_batch), enc_bb_batch

    def draw_bb(self, im, bbs, conf, color=(0,1,0)):
        new_im = np.copy(im)
        # print len(bbs)
        for i in range(len(bbs)):
            # print(bbs[i])
            if conf[i] >= 0.5:
                bb = self.bb_center2corner(bbs[i])
                # print bb
                cv2.rectangle(new_im, (int(bb[0][0] * im.shape[0]), int(bb[0][1] * im.shape[1])), 
                                      (int(bb[1][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                                      color, 2)
                cv2.putText(new_im,'{0:.2f}'.format(conf[i]), 
                                      (int(bb[0][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,1), 3, cv2.LINE_AA)
        return new_im

    # def draw_net_bb(self, im, bb_layer, scale):
    #     new_im = np.copy(im)
    #     print bb_layer.shape
    #     for p in range(bb_layer.shape[0]):
    #         for b in range(bb_layer.shape[1]):
    #             conf = bb_layer[p, b, 1]

    #             if p == 2: # conf > 0.5:
    #                 p_xy = (np.asarray([p % self.bb_size[scale]+1, p / self.bb_size[scale]+1], dtype=np.float32) + 0.5) * scale / self.im_size[0]
    #                 p_wh = self.bb_shapes[b] * 3. * scale / self.im_size[0]
    #                 default_bb = np.empty(4)
    #                 default_bb[:2] = p_xy
    #                 default_bb[2:] = p_wh
    #                 bb_c = default_bb + bb_layer[p, b, 2:6]
    #                 bb = self.bb_center2corner(bb_c)

    #                 cv2.rectangle(new_im, (int(bb[0][0] * im.shape[0]), int(bb[0][1] * im.shape[1])), 
    #                               (int(bb[1][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
    #                               (1,0,0), 2)
    #                 cv2.putText(new_im,'{0:.2f}'.format(conf), 
    #                               (int(bb[0][0] * im.shape[0]), int(bb[1][1] * im.shape[1])), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,1), 3, cv2.LINE_AA)
    #     return new_im

    def tf_IoU(self, bb1, bb2):
        pass

    def make_default_bb(self, scale):
        self.default_bb[scale] = []
        for b in range(self.num_bb):
            self.default_bb[scale].append([])

        for p_x in range(self.bb_size[scale]):
            for p_y in range(self.bb_size[scale]):
                # print p_x, p_y
                for b in range(self.num_bb):
                    p_xy = (np.asarray([p_x + 1, p_y + 1], dtype=np.float32) + 0.5) * scale / self.im_size[0]
                    # print p_xy
                    p_wh = self.bb_shapes[b] * 3. * scale / self.im_size[0]
                    d_bb = np.empty(4)
                    d_bb[:2] = p_xy
                    d_bb[2:] = p_wh
                    self.default_bb[scale][b].append(d_bb) # list of bb coordinates

        # for d in self.default_bb[scale][1]:
        #     print d

        # print len(self.default_bb[scale][1]), self.bb_size[scale]*self.bb_size[scale]

    def encode_2d_bb(self, gt_bbs, scale):
        # for scale in self.bb_size:
        # just 16 for now

        # print
        # print "================== encode_2d_bb"
        # print gt_bbs

        num_encoded = 0

        gt = np.zeros([self.bb_size[scale]]*2 + [self.out_channels])
        # print gt.shape
        for bb in gt_bbs: # for each gt bounding box

            for b in range(self.num_bb): # for each of the 3 aspect ratios
                for d_bb in self.default_bb[scale][b]: # for each bounding box (each pixel)
                
                    _, iou = self.iou(bb, d_bb)
                    p_x = int(np.round(d_bb[0] * (self.im_size[1] / scale) - 1.5))
                    p_y = int(np.round(d_bb[1] * (self.im_size[0] / scale) - 1.5))
                    if iou > 0.5:
                        num_encoded += 1
                        # print "over 0.5 "
                        # print bb
                        # print p_x, p_y, b, d_bb, iou
                        r_x = bb[0] - d_bb[0] # regression from default
                        r_y = bb[1] - d_bb[1]
                        r_w = bb[2] - d_bb[2]
                        r_h = bb[3] - d_bb[3]
                        # print [r_x, r_y, r_w, r_h]
                        gt[p_y, p_x, b*6 + 1] = 1.0
                        gt[p_y, p_x, b*6 + 2:b*6 + 6] = [r_x, r_y, r_w, r_h]


                        # bb1 = self.bb_center2corner(bb)
                        # bb2  = self.bb_center2corner(d_bb)
                        # c_x = r_x + d_bb[0]
                        # c_y = r_y + d_bb[1]
                        # c_w = r_w + d_bb[2]
                        # c_h = r_h + d_bb[3]
                        # bb3 = self.bb_center2corner([c_x, c_y, c_w, c_h])
                        # new_im = np.zeros((500,500, 3))
                        # print scale, iou
                        # cv2.rectangle(new_im, (int(bb1[0][0] * new_im.shape[0]), int(bb1[0][1] * new_im.shape[0])), 
                        #                       (int(bb1[1][0] * new_im.shape[0]), int(bb1[1][1] * new_im.shape[0])), 
                        #                       (0,1,0), 3)
                        # cv2.rectangle(new_im, (int(bb2[0][0] * new_im.shape[0]), int(bb2[0][1] * new_im.shape[0])), 
                        #                       (int(bb2[1][0] * new_im.shape[0]), int(bb2[1][1] * new_im.shape[0])), 
                        #                       (0,0,1), 3)
                        # cv2.rectangle(new_im, (int(bb3[0][0] * new_im.shape[0]), int(bb3[0][1] * new_im.shape[0])), 
                        #                       (int(bb3[1][0] * new_im.shape[0]), int(bb3[1][1] * new_im.shape[0])), 
                        #                       (1,0,0), 1)
                        # cv2.imshow("i", new_im)
                        # cv2.waitKey(0)


                    else:
                        gt[p_y, p_x, b*6 + 0] = 1.0 # is class 0 (background)


        # print
        # print gt[:,:,0*6+1]
        # print gt[:,:,1*6+1]
        # print gt[:,:,2*6+1]
        # print num_encoded
        # print "//================== encode_2d_bb"
        return gt

    def decode_2d_bb(self, encoded_bb, scale):
        # print "================== decode_2d_bb"

        # print encoded_bb[:,:,0*6+1]
        # print encoded_bb[:,:,1*6+1]
        # print encoded_bb[:,:,2*6+1]
        # print

        num_dencoded = 0 

        bbs = []
        confs = []
        # for x in range(encoded_bb.shape[1]):
        #     for y in range(encoded_bb.shape[0]):
        #         for b in range(3):
        #             if encoded_bb[y, x, b*6 + 1] > 0.5: # class 1 is 0.5 or more
        #                 [r_x, r_y, r_w, r_h] = encoded_bb[y, x, b*6 + 2:b*6 + 6]
        #                 c_x = r_x + d_bb[b][0]
        #                 c_y = r_y + d_bb[b][1]
        #                 c_w = r_w + d_bb[b][2]
        #                 c_h = r_h + d_bb[b][3]

        #                 bbs.append([c_x, c_y, c_w, c_h])
        #                 confs.append(encoded_bb[y, x, b*6 + 1])

        for b in range(self.num_bb):
            for d_bb in self.default_bb[scale][b]: # for each bounding box (each pixel)
                
            
                x = int(np.round(d_bb[0] * (self.im_size[1] / scale) - 1.5))
                y = int(np.round(d_bb[1] * (self.im_size[0] / scale) - 1.5))
                if encoded_bb[y, x, b*6 + 1] > 0.5: # class 1 is 0.5 or more
                    num_dencoded += 1
                    [r_x, r_y, r_w, r_h] = encoded_bb[y, x, b*6 + 2:b*6 + 6]

                    c_x = r_x + d_bb[0]
                    c_y = r_y + d_bb[1]
                    c_w = r_w + d_bb[2]
                    c_h = r_h + d_bb[3]

                    # print x, y, b, d_bb
                    # print [c_x, c_y, c_w, c_h] # [ 0.07578125  0.89375     0.0609375   0.09375   ]
                    # print [r_x, r_y, r_w, r_h] # [-0.23671875000000001, 0.70625000000000004, -0.22031249999999999, -0.1875]

                    soft = self.np_softmax(encoded_bb[y, x, b*6:b*6 + 1 + 1])
                    print encoded_bb[y, x, b*6:b*6 + 2], soft

                    bbs.append([c_x, c_y, c_w, c_h])
                    confs.append(soft[1])

                    # bb2  = self.bb_center2corner(d_bb)
                    # bb3 = self.bb_center2corner([c_x, c_y, c_w, c_h])
                    # new_im = np.zeros((500,500, 3))
                    # print x, y, b, encoded_bb[y, x, b*6 + 1]
                    # # cv2.rectangle(new_im, (int(bb1[0][0] * new_im.shape[0]), int(bb1[0][1] * new_im.shape[0])), 
                    # #                       (int(bb1[1][0] * new_im.shape[0]), int(bb1[1][1] * new_im.shape[0])), 
                    # #                       (0,1,0), 3)
                    # cv2.rectangle(new_im, (int(bb2[0][0] * new_im.shape[0]), int(bb2[0][1] * new_im.shape[0])), 
                    #                       (int(bb2[1][0] * new_im.shape[0]), int(bb2[1][1] * new_im.shape[0])), 
                    #                       (0,0,1), 3)
                    # cv2.rectangle(new_im, (int(bb3[0][0] * new_im.shape[0]), int(bb3[0][1] * new_im.shape[0])), 
                    #                       (int(bb3[1][0] * new_im.shape[0]), int(bb3[1][1] * new_im.shape[0])), 
                    #                       (1,0,0), 1)
                    # cv2.imshow("i", new_im)
                    # cv2.waitKey(0)

        # print num_dencoded
        # print "//================== decode_2d_bb"
        return bbs, confs

    def np_softmax(self, logits):
        z_exp = [np.exp(i) for i in logits]
        sum_z_exp = sum(z_exp)
        softmax = [i / sum_z_exp for i in z_exp]
        return softmax

    def loss(self, net_image, gt_image, scale):
        """
            for each bb - get IoU to all gt_bb
            IoU > 0.5 we call positive
                backprop classification + bb regression
            For all other, pick top k where conf loss is high
                backprop classificaiton
        """
        # find the bbox with best IoU and show it
        # start with just 16

        # reshape to a list
        print "====================================== Loss" + str(scale)

        vec_len = self.bb_size[scale]*self.bb_size[scale]

        net_vec = tf.reshape(net_image, [self.batch_size, vec_len, self.out_channels])
        gt_vec = tf.reshape(gt_image, [self.batch_size, vec_len, self.out_channels])
        print net_vec
        print gt_vec

        # split each default bounding box and concatenate into new lists - logits and bb
        net_logits = tf.slice(net_vec, [0,0,0], [self.batch_size, vec_len, 2])
        for b in range(1, self.num_bb):
            net_logits = tf.concat([net_logits,
                                  tf.slice(net_vec, [0,0,b*6], [self.batch_size, vec_len, 2])],
                                  axis=1)
        print net_logits

        net_bb = tf.slice(net_vec, [0,0,2], [self.batch_size, vec_len, 4])
        for b in range(1, self.num_bb):
            net_bb = tf.concat([net_bb,
                                  tf.slice(net_vec, [0,0,2+b*6], [self.batch_size, vec_len, 4])],
                                  axis=1)
        print net_bb

        gt_logits = tf.slice(gt_vec, [0,0,0], [self.batch_size, vec_len, 2])
        for b in range(1, self.num_bb):
            gt_logits = tf.concat([gt_logits,
                                  tf.slice(gt_vec, [0,0,b*6], [self.batch_size, vec_len, 2])],
                                  axis=1)
        print gt_logits

        gt_bb = tf.slice(gt_vec, [0,0,2], [self.batch_size, vec_len, 4])
        for b in range(1, self.num_bb):
            gt_bb = tf.concat([gt_bb,
                                  tf.slice(gt_vec, [0,0,2+b*6], [self.batch_size, vec_len, 4])],
                                  axis=1)
        print gt_bb

        cross_entropy_vec = tf.nn.softmax_cross_entropy_with_logits(logits=net_logits, labels=gt_logits)
        print cross_entropy_vec

        bb_L1_vec = tf.reduce_sum(tf.abs(net_bb - gt_bb), axis=2)
        print bb_L1_vec

        pos_mask = tf.cast(tf.equal(gt_logits[:,:,0], 0.), tf.float32) # where background logit is 0 (ie is not background)
        neg_mask = tf.cast(tf.equal(gt_logits[:,:,0], 1.), tf.float32) # where is background
        print pos_mask
        print neg_mask

        pos_xe_vec = tf.multiply(cross_entropy_vec, pos_mask)
        pos_bb_vec = tf.multiply(bb_L1_vec, pos_mask)
        print pos_xe_vec
        print pos_bb_vec

        neg_xe_vec = tf.multiply(cross_entropy_vec, neg_mask)
        neg_bb_vec = tf.multiply(bb_L1_vec, neg_mask)
        print neg_xe_vec
        print neg_bb_vec

        num_pos = tf.cast(tf.reduce_sum(pos_mask), tf.int32)
        num_neg = tf.minimum(tf.maximum(num_pos * 3, 3), vec_len) # take at least 3 negative
        self.debug[scale] = num_neg

        v_top_neg, i_top_neg = tf.nn.top_k(neg_xe_vec, k=num_neg)
        v_top_min = tf.reduce_min(v_top_neg)
        top_neg_mask = tf.cast(neg_xe_vec >= v_top_min, tf.float32)

        top_neg_xe_vec = tf.multiply(neg_xe_vec, top_neg_mask)
        top_neg_bb_vec = tf.multiply(neg_bb_vec, top_neg_mask)
        print top_neg_xe_vec
        print top_neg_bb_vec

        pos_loss = tf.reduce_mean(pos_xe_vec) + tf.reduce_mean(pos_bb_vec)
        print pos_loss

        neg_loss = tf.reduce_mean(top_neg_xe_vec) + tf.reduce_mean(top_neg_bb_vec)
        print neg_loss

        loss = pos_loss + neg_loss
        print loss
        print "/====================================== Loss" + str(scale)

        return loss

    def train(self):

        self.in_image = tf.placeholder(tf.float32, name = 'in_image',
                        shape= [self.batch_size] + self.im_size + [1])
        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        self.gt_image8 = tf.placeholder(tf.float32, name = 'gt_image8',
                        shape= [self.batch_size, self.bb_size[8], self.bb_size[8], self.out_channels])
        self.gt_image16 = tf.placeholder(tf.float32, name = 'gt_image16',
                        shape= [self.batch_size, self.bb_size[16], self.bb_size[16], self.out_channels])
        self.gt_image32 = tf.placeholder(tf.float32, name = 'gt_image32',
                        shape= [self.batch_size, self.bb_size[32], self.bb_size[32], self.out_channels])

        self.net = Network(self.in_image, self.im_size, self.num_bb, self.is_train)
        self.output8 = self.net.output_scale8
        self.output16 = self.net.output_scale16
        self.output32 = self.net.output_scale32

        self.total_loss = self.loss(self.output8, self.gt_image8, 8) \
                          + self.loss(self.output16, self.gt_image16, 16) \
                          + self.loss(self.output32, self.gt_image32, 32)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step1 = tf.train.AdamOptimizer(1e-3).minimize(self.total_loss)

        self.sess.run(tf.global_variables_initializer())

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.show_size[0]*4, self.show_size[1]*3)

        train_iterations = 1000
        start_time = dt.datetime.now()
        for i in range(train_iterations):
            ims, bbs, enc_bbs = self.get_batch()

            # d8,d16,d32 = self.sess.run([self.debug[8], self.debug[16], self.debug[32]], 
            #                                         feed_dict={self.in_image:ims, 
            #                                                     self.is_train:True,
            #                                                     self.gt_image8:enc_bbs[0],
            #                                                     self.gt_image16:enc_bbs[1],
            #                                                     self.gt_image32:enc_bbs[2]})
            # print d8,d16,d32

            _, loss, o8, o16, o32 = self.sess.run([train_step1, self.total_loss, self.output8, self.output16, self.output32],
                                                     feed_dict={self.in_image:ims, 
                                                                self.is_train:True,
                                                                self.gt_image8:enc_bbs[0],
                                                                self.gt_image16:enc_bbs[1],
                                                                self.gt_image32:enc_bbs[2]})

            if i % 10 == 0 or i == train_iterations-1:
                print i, "|", loss, "|", dt.datetime.now() - start_time
                show_im2 = self.make_show_image(ims[0], bbs[0], o8[0], o16[0], o32[0])
                cv2.imshow("Image", show_im2)
                cv2.waitKey(10)

    def make_show_image(self, image, gt_bbs, enc_bbs8, enc_bbs16, enc_bbs32):

        # print enc_bbs8.shape

        image = cv2.cvtColor(cv2.resize(image, self.show_size), cv2.COLOR_GRAY2RGB)
        conf_gt = np.ones(len(gt_bbs), np.float32)
        image_gt = self.draw_bb(image, gt_bbs, conf_gt, (0,1,1))

        gt_enc_bbs8 = self.encode_2d_bb(gt_bbs, 8)
        gt_enc_bbs16 = self.encode_2d_bb(gt_bbs, 16)
        gt_enc_bbs32 = self.encode_2d_bb(gt_bbs, 32)

        decoded8, confs8 = self.decode_2d_bb(gt_enc_bbs8, 8)
        gt_image_dec0 = self.draw_bb(image, decoded8, confs8)
        decoded16, confs16 = self.decode_2d_bb(gt_enc_bbs16, 16)
        gt_image_dec = self.draw_bb(image, decoded16, confs16)
        decoded32, confs32 = self.decode_2d_bb(gt_enc_bbs32, 32)
        gt_image_dec2 = self.draw_bb(image, decoded32, confs32)
        gt_image_dec_total = self.draw_bb(image, decoded8 + decoded16 + decoded32, confs8 + confs16 + confs32, (1,1,0))

        decoded8, confs8 = self.decode_2d_bb(enc_bbs8, 8)
        image_dec0 = self.draw_bb(image, decoded8, confs8)
        decoded16, confs16 = self.decode_2d_bb(enc_bbs16, 16)
        image_dec = self.draw_bb(image, decoded16, confs16)
        decoded32, confs32 = self.decode_2d_bb(enc_bbs32, 32)
        image_dec2 = self.draw_bb(image, decoded32, confs32)
        image_dec_total = self.draw_bb(image, decoded8 + decoded16 + decoded32, confs8 + confs16 + confs32, (1,1,0))

        spacer = np.zeros((self.show_size[0], self.show_size[1], 3))
        show_im = np.concatenate([image, image_gt, gt_image_dec_total, image_dec_total], axis=1)
        show_im_gt_scales = np.concatenate([spacer, gt_image_dec0, gt_image_dec, gt_image_dec2], axis=1)
        show_im_scales = np.concatenate([spacer, image_dec0, image_dec, image_dec2], axis=1)
        show_im = np.concatenate([show_im, show_im_gt_scales, show_im_scales])

        return show_im

    def show_images(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.show_size[0]*3, self.show_size[1]*2)

        for i in range(100):
            ims, bbs, enc_bbs = self.get_batch()

            print i
            show_im = self.make_show_image(ims[0], bbs[0], enc_bbs[0][0], enc_bbs[1][0], enc_bbs[2][0])
            cv2.imshow("Image", show_im)
            cv2.waitKey(1000)
            


if __name__ == "__main__":
    with tf.Session() as sess:  
        trainer = MNIST_OD_Trainer(sess)

        trainer.show_images()
        # trainer.train()
