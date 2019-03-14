#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by GoogLeNet_InceptionV1 on 19-3-14


import tensorflow as tf
import tensorflow.contrib.slim as slim


def inception_moudle_v1(net,scope,filters_num):
    with tf.variable_scope(scope):
        with tf.variable_scope('bh1'):
            bh1 = slim.conv2d(net,filters_num[0],1,scope='bh1_conv1_1x1')
        with tf.variable_scope('bh2'):
            bh2 = slim.conv2d(net,filters_num[1],1,scope='bh2_conv1_1x1')
            bh2 = slim.conv2d(bh2,filters_num[2],3,scope='bh2_conv2_3x3')
        with tf.variable_scope('bh3'):
            bh3 = slim.conv2d(net,filters_num[3],1,scope='bh3_conv1_1x1')
            bh3 = slim.conv2d(bh3,filters_num[4],5,scope='bh3_conv2_5x5')
        with tf.variable_scope('bh4'):
            bh4 = slim.max_pool2d(net,3,scope='bh4_max_3x3')
            bh4 = slim.conv2d(bh4,filters_num[5],1,scope='bh4_conv_1x1')
        net = tf.concat([bh1,bh2,bh3,bh4],axis=3)
    return net



def V1_slim(inputs,num_cls,is_train = False,keep_prob=0.4,spatital_squeeze=True):
    with tf.name_scope('reshape'):
        net = tf.reshape(inputs, [-1, 224, 224, 3])

    with tf.variable_scope('GoogLeNet_V1'):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(5e-4),
                weights_initializer=slim.xavier_initializer(),
        ):
            with slim.arg_scope(
                [slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                    padding='SAME',
                    stride=1,
            ):
                net = slim.conv2d(net,64,7,stride=2,scope='layer1')
                net = slim.max_pool2d(net,3,stride=2,scope='layer2')
                net = tf.nn.lrn(net)
                net = slim.conv2d(net,64,1,scope='layer3')
                net = slim.conv2d(net,192,3,scope='layer4')
                net = tf.nn.lrn(net)
                net = slim.max_pool2d(net,3,stride=2,scope='layer5')
                net = inception_moudle_v1(net,'layer6',[64,96,128,16,32,32])
                net = inception_moudle_v1(net,'layer8',[128,128,192,32,96,64])
                net = slim.max_pool2d(net,3,stride=2,scope='layer10')
                net = inception_moudle_v1(net,'layer11',[192,96,208,16,48,64])
                net = inception_moudle_v1(net,'layer13',[160,112,224,24,64,64])
                net_1 = net
                net = inception_moudle_v1(net,'layer15',[128,128,256,24,64,64])
                net = inception_moudle_v1(net,'layer17',[112,144,288,32,64,64])
                net_2 = net
                net = inception_moudle_v1(net,'lauer19',[256,160,320,32,128,128])
                net = slim.max_pool2d(net,3,stride=2,scope='layer21')
                net = inception_moudle_v1(net,'layer22',[256,160,320,32,128,128])
                net = inception_moudle_v1(net,'layer24',[384,192,384,48,128,128])

                net = slim.avg_pool2d(net,7,stride=1,padding='VALID',scope='layer26')
                net = slim.dropout(net,keep_prob=keep_prob,scope='dropout')
                net = slim.conv2d(net,num_cls,1,activation_fn=None, normalizer_fn=None,scope='layer27')
                if spatital_squeeze:
                    net = tf.squeeze(net,[1,2],name='squeeze')
                net = slim.softmax(net,scope='softmax2')

                if is_train:
                    net_1 = slim.avg_pool2d(net_1, 5, padding='VALID', stride=3, scope='auxiliary0_avg')
                    net_1 = slim.conv2d(net_1, 128, 1, scope='auxiliary0_conv_1X1')
                    net_1 = slim.flatten(net_1)
                    net_1 = slim.fully_connected(net_1,1024,scope='auxiliary0_fc1')
                    net_1 = slim.dropout(net_1, 0.7)
                    net_1 = slim.fully_connected(net_1,num_cls,activation_fn=None,scope='auxiliary0_fc2')
                    net_1 = slim.softmax(net_1, scope='softmax0')

                    net_2 = slim.avg_pool2d(net_2, 5, padding='VALID', stride=3, scope='auxiliary1_avg')
                    net_2 = slim.conv2d(net_2, 128, 1, scope='auxiliary1_conv_1X1')
                    net_2 = slim.flatten(net_2)
                    net_2 = slim.fully_connected(net_2,1024,scope='auxiliary1_fc1')
                    net_2 = slim.dropout(net_2, 0.7)
                    net_2 = slim.fully_connected(net_2,num_cls,activation_fn=None,scope='auxiliary1_fc2')
                    net_2 = slim.softmax(net_2, scope='softmax1')

                    net = net_1 * 0.3 + net_2 * 0.3 + net * 0.4
                    print(net.shape)

    return net

class testInceptionV1(tf.test.TestCase):
    def testBuildClassifyNetwork(self):
        inputs = tf.random_uniform((5,224,224,3))
        logits = V1_slim(inputs,10)
        print(logits)

if __name__ == '__main__':
    tf.test.main()