#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Inception on 19-1-23

import tensorflow as tf
import tensorflow.contrib.slim as slim


def inception_moudle_v2(net,scope,filters_num,pool_type,stride):
    with tf.variable_scope(scope):
        if filters_num[0] != 0:
            with tf.variable_scope('bh1'):
                bh1 = slim.conv2d(net,filters_num[0],1,stride=stride,scope='bh1_conv1_1x1')
        with tf.variable_scope('bh2'):
            bh2 = slim.conv2d(net,filters_num[1],1,stride=1,scope='bh2_conv1_1x1')
            bh2 = slim.conv2d(bh2,filters_num[2],3,stride=stride,scope='bh2_conv2_3x3')
        with tf.variable_scope('bh3'):
            bh3 = slim.conv2d(net,filters_num[3],1,stride=1,scope='bh3_conv1_1x1')
            bh3 = slim.conv2d(bh3,filters_num[4],3,stride=1,scope='bh3_conv2_3x3')
            bh3 = slim.conv2d(bh3,filters_num[5],3,stride=stride,scope='bh3_conv3_3x3')
        with tf.variable_scope('bh4'):
            if pool_type == 'avg':
                bh4 = slim.avg_pool2d(net,3,stride=stride,scope='bh4_avg_3x3')
            elif pool_type == 'max':
                bh4 = slim.max_pool2d(net,3,stride=stride, scope='bh4_max_3x3')
            else:
                raise TypeError("没有此参数类型（params valid）")
            if filters_num[0] != 0:
                bh4 = slim.conv2d(bh4,filters_num[6],1,stride=1,scope='bh4_conv_1x1')
                net = tf.concat([bh1,bh2,bh3,bh4],axis=3)
            else:
                net = tf.concat([bh2,bh3,bh4],axis=3)
    return net

def V2_slim(inputs, num_cls, keep_prob=0.8,is_training = False, spatital_squeeze = True):
    batch_norm_params = {
        'decay': 0.998,
        'epsilon': 0.001,
        'scale': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }

    net = inputs
    with tf.name_scope('reshape'):
        net = tf.reshape(net,[-1,224,224,3])

    with tf.variable_scope('GoogLeNet_V2'):
            with slim.arg_scope(
                [slim.conv2d,slim.separable_conv2d],
                weights_initializer=slim.xavier_initializer(),
                normalizer_fn= slim.batch_norm,
                normalizer_params= batch_norm_params,
                # normalizer_fn = tf.layers.batch_normalization,
                # normalizer_params = params
            ):
                with slim.arg_scope(
                  [slim.conv2d,slim.max_pool2d,slim.avg_pool2d,slim.separable_conv2d],
                    stride=1,
                    padding='SAME'
                ):
                    with slim.arg_scope([slim.batch_norm],**batch_norm_params):
                        net = slim.separable_conv2d(net, 64, 7, depth_multiplier=8, stride=2,
                                                    weights_initializer=slim.xavier_initializer(), scope='layer1')
                        net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='layer2')
                        net = slim.conv2d(net, 64, 1, stride=1, padding='SAME', scope='layer3')
                        net = slim.conv2d(net, 192, 3, stride=1, padding='SAME', scope='layer4')
                        net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='layer5')
                        net = inception_moudle_v2(net,scope='layer6_3a',filters_num=[64, 64, 64, 64, 96, 96, 32],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer9_3b',filters_num=[64, 64, 96, 64, 96, 64, 64],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer12_3c',filters_num=[0, 128,160,64, 96, 96],pool_type='max',stride=2)
                        net = inception_moudle_v2(net,scope='layer15_4a',filters_num=[224,64,96,96,128,128,128],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer18_4b',filters_num=[192,96,128,96,128,128,128],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer21_4c',filters_num=[160,128,160,128,160,160,128],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer24_4d',filters_num=[96,128,192,160,192,192,128],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer27_4e',filters_num=[0,128,192,192,256,256],pool_type='max',stride=2)
                        net = inception_moudle_v2(net,scope='layer30_5a',filters_num=[352,192,320,160,224,224,128],pool_type='avg',stride=1)
                        net = inception_moudle_v2(net,scope='layer33_5b',filters_num=[352,192,320,192,224,224,128],pool_type='max',stride=1)

                        net = slim.avg_pool2d(net,7,stride=1,padding='VALID',scope="layer36_avg")

                        net = slim.dropout(net,keep_prob=keep_prob,scope="dropout")
                        net = slim.conv2d(net,num_cls,1,activation_fn=None,normalizer_fn=None,scope="layer37")
                        if spatital_squeeze:
                            net = tf.squeeze(net,[1,2],name='squeeze')
                        net = slim.softmax(net,scope="softmax")

                        return net





# class testInceptionV1(tf.test.TestCase):
#     def testBuildClassifyNetwork(self):
#         inputs = tf.random_uniform((5,224,224,3))
#         logits = V1_slim(inputs,10)
#         print(logits)
#
# class testInceptionTiny(tf.test.TestCase):
#     def testBuildNet(self):
#         inputs = tf.random_uniform((3,32,32,3))
#         logits = Inception_V1_tiny(inputs,num_cls=10)
#         print(logits)

#




if __name__ == '__main__':

    # tf.test.main()
    #
    input = tf.placeholder(tf.float32, [None, 32, 32, 3])

    # train_net = Inception_V1(input,num_cls=2)
    logits = V2_slim(input, num_cls=10, is_training=True)
    # logits = Inception_V1(input,num_cls=10)
    import numpy as np

    # one = np.array([1,0,3,2])
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())


        test = sess.run(tf.random_uniform((5, 32, 32, 3)))
        # res = sess.run(train_net, feed_dict={input:test})
    #     # print(res)
        res = sess.run(logits,feed_dict={input:test})
        print(res)
    #
    #
    #     tensor = sess.run(tf.random_uniform((2,3,5)))
    #     print(tensor)
    #     res = sess.run(tf.argmax(tensor,1))
    #     print(res)
    #     print(sess.run(tf.shape(res)))
    #
    #     tensor_one = tf.convert_to_tensor(one)
    #     print(tensor_one.eval())
    #     print(sess.run(tf.one_hot(tensor_one,depth=4)))
    #     one_hot_res = tf.one_hot(tensor_one,depth=4)
    #     print(sess.run(tf.argmax(one_hot_res,0)))
