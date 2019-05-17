#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by model on 19-4-17

import tensorflow as tf
import tensorflow.contrib.slim as slim
import net.Detection.YOLOV1.config as cfg
import numpy as np




class YOLO_Net(object):
    def __init__(self,is_pre_training=False,is_training = True):
        self.classes = cfg.VOC07_CLASS
        self.pre_train_num = cfg.PRE_TRAIN_NUM
        self.det_cls_num = len(self.classes)
        self.image_size = cfg.DET_IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.PER_CELL_CHECK_BOXES
        self.output_size = (self.cell_size * self.cell_size) * ( 5 * self.boxes_per_cell + self.det_cls_num)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.det_cls_num
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        self.object_scale = cfg.OBJ_CONFIDENCE_SCALE
        self.no_object_scale = cfg.NO_OBJ_CONFIDENCE_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.learning_rate = 0.0001
        self.batch_size = cfg.BATCH_SIZE
        self.keep_prob = cfg.KEEP_PROB
        self.pre_training = is_pre_training

        self.offset = np.transpose(
            np.reshape(
                np.array(
                    [np.arange(self.cell_size)]*self.cell_size*self.boxes_per_cell
                ),(self.boxes_per_cell,self.cell_size,self.cell_size)
            ),(1,2,0)
        )

        self.bn_params = cfg.BATCH_NORM_PARAMS
        self.is_training = tf.placeholder(tf.bool)
        if self.pre_training:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='images')
        else:
            self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')

        self.logits = self.build_network(self.images,is_training=self.is_training)

        if is_training:
            if self.pre_training:
                self.labels = tf.placeholder(tf.float32, [None,self.pre_train_num])
                self.classify_loss(self.logits,self.labels)
                self.total_loss = tf.losses.get_total_loss()
                self.evalution = self.classify_evalution(self.logits,self.labels)
                print('预训练网络')
            else:
                self.labels = tf.placeholder(tf.float32, [None,self.cell_size,self.cell_size,5+self.det_cls_num])
                self.det_loss_layer(self.logits,self.labels)
                self.total_loss = tf.losses.get_total_loss()
                tf.summary.scalar('total_loss', self.total_loss)
                print('识别网络')




    def build_network(self, images,is_training = True,scope = 'yolov1'):
        net = images
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.00004)):
                with slim.arg_scope([slim.conv2d],
                                    weights_initializer=slim.xavier_initializer(),
                                    normalizer_fn=slim.batch_norm,
                                    activation_fn=slim.nn.leaky_relu,
                                    normalizer_params=self.bn_params):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                        net = slim.conv2d(net, 64, [7, 7], stride=2, padding='SAME', scope='layer1')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

                        net = slim.conv2d(net, 192, [3, 3], stride=1, padding='SAME', scope='layer2')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')

                        net = slim.conv2d(net, 128, [1, 1], stride=1, padding='SAME', scope='layer3_1')
                        net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='layer3_2')
                        net = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='layer3_3')
                        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='layer3_4')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')

                        net = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='layer4_1')
                        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='layer4_2')
                        net = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='layer4_3')
                        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='layer4_4')
                        net = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='layer4_5')
                        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='layer4_6')
                        net = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='layer4_7')
                        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='layer4_8')
                        net = slim.conv2d(net, 512, [1, 1], stride=1, padding='SAME', scope='layer4_9')
                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer4_10')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool4')

                        net = slim.conv2d(net, 512, [1, 1], stride=1, padding='SAME', scope='layer5_1')
                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer5_2')
                        net = slim.conv2d(net, 512, [1, 1], stride=1, padding='SAME', scope='layer5_3')
                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer5_4')

                        if self.pre_training:
                            net = slim.avg_pool2d(net, [7, 7], stride=1, padding='VALID', scope='clssify_avg5')
                            net = slim.flatten(net)
                            net = slim.fully_connected(net, self.pre_train_num, activation_fn=slim.nn.leaky_relu,
                                                       scope='classify_fc1')
                            return net

                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer5_5')
                        net = slim.conv2d(net, 1024, [3, 3], stride=2, padding='SAME', scope='layer5_6')

                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer6_1')
                        net = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='layer6_2')

                        net = slim.flatten(net)

                        net = slim.fully_connected(net, 1024, activation_fn=slim.nn.leaky_relu, scope='fc1')
                        net = slim.dropout(net, 0.5)
                        net = slim.fully_connected(net, 4096, activation_fn=slim.nn.leaky_relu, scope='fc2')
                        net = slim.dropout(net, 0.5)
                        net = slim.fully_connected(net, self.output_size, activation_fn=None, scope='fc3')
                        # N, 7,7,30
                        # net = tf.reshape(net,[-1,S,S,B*5+C])
            return net

    def classify_loss(self,logits,labels):
        with tf.name_scope('classify_loss') as scope:
            _loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            mean_loss = tf.reduce_mean(_loss)
            tf.losses.add_loss(mean_loss)
            tf.summary.scalar(scope + 'classify_mean_loss', mean_loss)

    def classify_evalution(self,logits,labels):
        with tf.name_scope('classify_evaluation') as scope:
            correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accurary = tf.reduce_mean(tf.cast(correct_pre, 'float'))
            # tf.summary.scalar(scope + 'accuracy:', accurary)
        return accurary


    '''
    @:param predicts shape->[N,7x7x30]
    @:param labels   shape->[N,7,7,25]  <==>[N,h方向,w方向,25] ==>[N,7,7,25(1:是否负责检测,2-5:坐标,6-25:类别one-hot)]
    '''

    def det_loss_layer(self, predicts, labels, scope='det_loss'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                         [-1, 7, 7, 20])  # 类别预测 ->[batch_size,cell_size,cell_size,num_cls]
            predict_scale = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                       [-1, 7, 7, 2])  # 置信率预测-> [batch_size,cell_size,cell_size,boxes_per_cell]
            predict_boxes = tf.reshape(predicts[:,self.boundary2:],
                                       [-1, 7, 7, 2, 4])  # 坐标预测->[batch_size,cell_size,cell_size,boxes_per_cell,4]

            response = tf.reshape(labels[:, :, :, 0], [-1, 7, 7, 1])  # 标签置信率，用来判断cell是否负责检测
            boxes = tf.reshape(labels[:, :, :, 1:5], [-1, 7, 7, 1, 4])  # 标签坐标
            boxes = tf.tile(boxes,
                            [1, 1, 1, 2, 1]) / self.image_size  # 标签坐标，由于预测是2个，因此需要将标签也变成2个，同时对坐标进行yolo形式归一化
            classes = labels[:, :, :, 5:]  # 标签类别

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, 7, 7, 2])
            offset = tf.tile(offset, [tf.shape(boxes)[0], 1, 1, 1])
            predict_boxes_tran = tf.stack([
                1. * (predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                1. * (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                tf.square(predict_boxes[:, :, :, :, 2]),
                tf.square(predict_boxes[:, :, :, :, 3])
            ], axis=-1)
            # predict_boxes_tran = tf.transpose(predict_boxes_tran,[1,2,3,4,0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
            no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
            boxes_tran = tf.stack([
                1. * boxes[:, :, :, :, 0] * 7 - offset,
                1. * boxes[:, :, :, :, 1] * 7 - tf.transpose(offset, (0, 2, 1, 3)),
                tf.sqrt(boxes[:, :, :, :, 2]),
                tf.sqrt(boxes[:, :, :, :, 3])
            ], axis=-1)

            # 类别损失
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                        name='class_loss') * self.class_scale

            # 对象损失
            object_delta = object_mask * (predict_scale - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss') * self.object_scale

            # 无对象损失
            no_object_delta = no_object_mask * predict_scale
            no_object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(no_object_delta), axis=[1, 2, 3]),
                                            name='no_object_loss') * self.no_object_scale

            # 坐标损失
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                        name='coord_loss') * self.coord_scale
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(no_object_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', no_object_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)


    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
               Args:
                 boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
                 boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
               Return:
                 iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
               """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0], axis=-1)
            # boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0], axis=-1)
            # boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


if __name__ == '__main__':
    # tf.test.main()
    inputs = tf.placeholder(tf.float32,[None,448,448,3])
    # logits = YoLoNetModel_slim(inputs,is_training=False)
    yolo = YOLO_Net(is_pre_training=True)
    logits = yolo.logits

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(tf.random_uniform((3,224,224,3)))
        res = sess.run(logits,feed_dict={yolo.images:val,yolo.is_training:False})
        print(res)
        print(sess.run(tf.shape(res)))