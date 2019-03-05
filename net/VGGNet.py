#!/usr/bin/env python3.6.3
# encoding: utf-8
# @Time    : 2018/12/5 1:11
# @Author  : Echo
# set(  = "https://blog.csdn.net/shankezh" )
# @contact: cloud_happy@163.com
# @Site    :
# @File    : AlexNet.py
# @Software: PyCharm


import tensorflow as tf
import tensorflow.contrib.slim as slim

def VGG16_slim(inputs,num_cls,vgg_mean,keep_prob=0.5):
    net = inputs

    # net = tf.cast(net, tf.float32) / 255.  # 转换数据类型并归一化

    # with tf.name_scope('reshpae'):
    #     net = tf.reshape(net,[-1,224,224,3])
    with tf.variable_scope('vgg_net'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            # weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            # weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.constant_initializer(0.0)
                            ):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='same',
                                stride=1):
                net = slim.repeat(net,2,slim.conv2d,64,[3,3],scope='conv1')
                net = slim.max_pool2d(net,[2,2],stride=2,scope='maxpool1')

                net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
                net = slim.max_pool2d(net,[2,2],stride=2,scope='maxpool2')

                net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
                net = slim.max_pool2d(net,[2,2],stride=2,scope='maxpool3')

                net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
                net = slim.max_pool2d(net,[2,2],stride=2,scope='maxpool4')

                net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
                net = slim.max_pool2d(net,[2,2],stride=2,scope='maxpool5')

                net = slim.flatten(net,scope='flatten')

                net = slim.stack(net,slim.fully_connected,[1024,1024,num_cls],scope='fc')
                net = slim.softmax(net,scope='softmax')
        return net


class testVggNet(tf.test.TestCase):
    def testBuildClassifyNetwork(self):
        inputs = tf.random_uniform((5,224,224,3))
        logits = VGG16_slim(inputs,10)
        print(logits)



if __name__ == '__main__':
    tf.test.main()