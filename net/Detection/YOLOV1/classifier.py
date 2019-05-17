#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by classifer on 19-5-7
import net.Detection.YOLOV1.config as cfg
from net.Detection.YOLOV1.model import YOLO_Net
import tensorflow as tf
import numpy as np
from coms.utils import isHasGpu
import cv2
import tensorflow.contrib.slim as slim

class Classifier(object):
    def __init__(self,net):
        self.net = net
        self.model_cls_dir = cfg.CLS_MODEL_DIR
        if isHasGpu():
            gpu_option = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_option)
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)

    def test(self):
        # self.load_model()
        self.load_part_model()
        cls_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                    'truck']
        path = r'../../tf_file/NetModel/Test/Cifar10/'
        for i in range(1, 11):
            name = str(i) + '.jpg'
            img = cv2.imread(path + name)
            # img = cv2.resize(img, (32, 32))
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.
            img = np.array([img])
            res = self.sess.run(self.net.logits, feed_dict={self.net.images: img, self.net.is_training: False})

            # print(res)
            print('{}.jpg detect result is : '.format(str(i)) + cls_list[np.argmax(res)])

    def load_model(self):
        model_file = tf.train.latest_checkpoint(self.model_cls_dir)
        self.saver.restore(self.sess, model_file)

    def load_part_model(self):
        model_vars = slim.get_model_variables()
        model_file = tf.train.latest_checkpoint(self.model_cls_dir)
        reader = tf.train.NewCheckpointReader(model_file)
        dic = reader.get_variable_to_shape_map()
        print(dic)
        # for var in dic:
        #     print(self.sess.run(var))


        exclude = ['yolov1/classify_fc1/weights', 'yolov1/classify_fc1/biases']
        # vars_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # self.saver = tf.train.Saver(vars_to_restore)

        vars_restore_map = {}
        for var in model_vars:
            if var.op.name in dic and var.op.name not in exclude:
                vars_restore_map[var.op.name] = var

        self.saver = tf.train.Saver(vars_restore_map)
        self.saver.restore(self.sess,model_file)




if __name__ == '__main__':
    yolo = YOLO_Net(is_pre_training=True)
    classifier = Classifier(yolo)
    classifier.test()