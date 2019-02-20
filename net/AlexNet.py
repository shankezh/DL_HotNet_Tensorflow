#!/usr/bin/env python3.6.3
# encoding: utf-8
# @Time    : 2018/12/5 1:11
# set(  = "https://blog.csdn.net/shankezh" )
# @contact: cloud_happy@163.com
# @Site    : 
# @File    : AlexNet.py
# @Software: PyCharm


import tensorflow as tf
import tensorflow.contrib.slim as slim




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#
# def conv2d(x,W):
#     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def conv2d(x,W,stride,pad):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=pad)

def max_pool(x,ksize,stride,pad):
    return tf.nn.max_pool(x, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding=pad)

# 归一化
def norm(x, depth_radius=5.0, bias=1.0, alpha=1.0, beta=0.5):
    return tf.nn.lrn(x, depth_radius=depth_radius, bias=bias, alpha=alpha,beta=beta)



x = tf.placeholder(tf.float32,[None,227*227])
y_ = tf.placeholder(tf.float32,[None,2])
lr = 0.001


'''
模型AlexNet

卷积层：5层
全连接层：3层
深度：8层
参数个数：60M 
神经元个数：650k
分类数目：1000类
'''


def AlexNetModel_slim(inputs,num_cls=2,keep_prob = 0.5):
        with tf.name_scope('reshape'):
            inputs = tf.reshape(inputs,[-1,227,227,3])

        with tf.variable_scope('alex_net'):
            with slim.arg_scope([slim.conv2d,slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                                    padding = 'same',
                                    stride = 1):
                    net = slim.conv2d(inputs,96,11,stride=4,padding='VALID')

                    net = norm(net,depth_radius=5.0,bias=2.0,alpha=1e-4,beta=0.75)
                    net = slim.max_pool2d(net,3,stride=2,padding='VALID')

                    net = slim.conv2d(net,256,5)
                    net = norm(net,depth_radius=5.0,bias=2.0,alpha=1e-4,beta=0.75)
                    net = slim.max_pool2d(net,3,stride=2,padding='VALID')

                    net = slim.conv2d(net,384,3)

                    net = slim.conv2d(net,384,3)

                    net = slim.conv2d(net,256,3)
                    met = slim.max_pool2d(net,3,stride=2,padding='VALID')

                    net = slim.flatten(net)

                    net = slim.fully_connected(net,4096)

                    net = slim.dropout(net,keep_prob=keep_prob)

                    net = slim.fully_connected(net,4096)

                    net = slim.dropout(net,keep_prob=keep_prob)

                    net = slim.fully_connected(net,num_cls)
        return net



def AlexNetModel(input_x):

    num_cls = 2 # 定义需要分类数量
    keep_prob = tf.placeholder(tf.float32)

    with  tf.name_scope('reshape'):
        x_img = tf.reshape(input_x,[-1,227,227,3])

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([11,11,3,96])
        b_conv1 = bias_variable([96])
        hc1 = tf.nn.relu(conv2d(x_img,w_conv1,stride=4,pad="VALID") + b_conv1)
        print('hc1:', hc1.shape)

    with tf.name_scope('norm1'):
        norm1 = norm(hc1,depth_radius=5.0, bias=2.0, alpha=1e-4,beta=0.75)
        print('norm1:',norm1.shape)


    with tf.name_scope('pool1'):
        # pool1 = max_pool_2x2(hc1)
        pool1 = max_pool(norm1,ksize=3,stride=2,pad='VALID')
        print('pool1' ,pool1.shape)



    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5,5,96,256])
        b_conv2 = bias_variable([256])
        hc2 = tf.nn.relu(conv2d(pool1,w_conv2,stride=1,pad='SAME') + b_conv2)
        print('hc2:', hc2.shape)


    with tf.name_scope('norm2'):
        norm2 = norm(hc2,depth_radius=5.0,bias=2.0,alpha=1e-4,beta=0.75)
        print('norm2:',norm2.shape)

    with tf.name_scope('pool2'):
        pool2 = max_pool(norm2,ksize=3,stride=2,pad='VALID')
        print('pool2:',pool2.shape)

    with tf.name_scope('conv3'):
        w_conv3 = weight_variable([3,3,256,384])
        b_conv3 = bias_variable([384])
        hc3 = tf.nn.relu(conv2d(pool2, w_conv3, stride=1, pad='SAME') + b_conv3)
        print('hc3:', hc3.shape)

    with tf.name_scope('conv4'):
        w_conv4 = weight_variable([3,3,384,384])
        b_conv4 = bias_variable([384])
        hc4 = tf.nn.relu(conv2d(hc3, w_conv4, stride=1, pad='SAME') + b_conv4)
        print('hc4:', hc4.shape)


    with tf.name_scope('conv5'):
        w_conv5 = weight_variable([3,3,384,256])
        b_conv5 = bias_variable([256])
        hc5 = tf.nn.relu(conv2d(hc4, w_conv5, stride=1, pad='SAME') + b_conv5)
        print('hc5:', hc5.shape)


    with tf.name_scope('pool5'):
        pool5 = max_pool(hc5, ksize=3, stride=2, pad='VALID')
        print('pool5:', pool5.shape)


    with tf.name_scope('flatten'):
        flat = tf.reshape(pool5, [-1,6*6*256])
        print('flat:', flat.shape)


    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([6*6*256,4096])
        b_fc1 = bias_variable([4096])
        fc1 = tf.nn.relu(tf.matmul(flat,w_fc1) + b_fc1)
        print('fc1:', fc1.shape)


    with tf.name_scope('dropout1'):
        fc1_drop = tf.nn.dropout(fc1,keep_prob=keep_prob)


    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([4096,4096])
        b_fc2 = bias_variable([4096])
        fc2 = tf.nn.relu(tf.matmul(fc1_drop,w_fc2) + b_fc2)
        print('fc2:', fc2.shape)


    with tf.name_scope('dropout2'):
        fc2_drop = tf.nn.dropout(fc2,keep_prob=keep_prob)

    with tf.name_scope('fc3'):
        w_fc3 = weight_variable([4096,num_cls])
        b_fc3 = bias_variable([num_cls])
        y_conv = tf.nn.softmax(tf.matmul(fc2_drop,w_fc3) + b_fc3)


    return y_conv,keep_prob


def loss(y_conv,y_):
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv,labels=y_))
    return cost

def optimizer(lr,cost):
    with tf.name_scope('optimizer'):
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    return optim

def evaluation(y_conv,y_):
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


class testAlexnet(tf.test.TestCase):
    def testBuildClassifyNetwork(self):
        inputs = tf.random_uniform((5,227,227,3))
        logits = AlexNetModel_slim(inputs)
        print(logits)




if __name__ == '__main__':
    tf.test.main()