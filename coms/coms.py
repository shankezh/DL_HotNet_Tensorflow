#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Commons on 19-1-30


import tensorflow as tf
import tensorflow.contrib.slim as slim

# 载入模型
def load_weights(self):
    pass

# 训练模型
def train_model(self):
    pass

# 微调模型
def fine_tune(self):
    pass

# 预测
def predict(self):
    pass



# 带BN的训练函数
def optimizer_bn(lr,loss,mom=0.9,fun = 'mm'):
    with tf.name_scope('optimzer_bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("BN parameters: ", update_ops)
        with tf.control_dependencies([tf.group(*update_ops)]):
            optim = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
            train_op = slim.learning.create_train_op(loss,optim)
    return train_op

# 训练函数
def optimizer(lr,loss,mom=0.9,fun = 'mm'):
    with tf.name_scope('optimizer'):
        if fun == 'mm':
            optim = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9).minimize(loss=loss)
        elif fun == 'gdo':
            optim = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss=loss)
        elif fun == 'adam':
            optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss= loss)
        else:
            raise TypeError('未输入正确训练函数')
    return optim



# 误差
def loss(logits, labels, fun='cross'):
    with tf.name_scope('loss') as scope:
        if fun == 'cross':
            _loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            mean_loss = tf.reduce_mean(_loss)
        else:
            raise TypeError('未输入正确误差函数')
        tf.summary.scalar(scope + 'mean_loss', mean_loss)
    return mean_loss


'''
准确率计算,评估模型
由于测试存在切片测试合并问题，因此正确率的记录放到了正式代码中
'''
def evaluation(logits,labels):
    with tf.name_scope('evaluation') as scope:
        correct_pre = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        accurary = tf.reduce_mean(tf.cast(correct_pre,'float'))
        tf.summary.scalar(scope + 'accuracy:', accurary)

    return accurary



if __name__ == '__main__':
    pass