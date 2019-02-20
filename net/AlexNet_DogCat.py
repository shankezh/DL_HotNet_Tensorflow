#!/usr/bin/env python3.6.3
# encoding: utf-8
# @Time    : 2018/12/27 23:19
# set(  = "https://blog.csdn.net/shankezh" )
# @contact: cloud_happy@163.com
# @File    : AlexNet_DogCat.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import cv2
import numpy as np
import tensorflow as tf

import coms.coms as coms
import coms.utils as utils
import coms.pre_process as pre_pro
import net.AlexNet as AlexNet
import time



def main():
    train_dir = ''
    test_dir = ''
    model_dir = ''
    logdir = ''
    num_cls = 2
    is_train = True
    is_load_model = True
    BATCH_SZIE = 100
    EPOCH_NUM = 20
    ITER_NUM = 20000  # 60000 / 50
    KEEP_PROB = 0.5

    if utils.isLinuxSys():
        train_dir = r'/home/zhuhao/DataSets/CatAndDog/train'
        test_dir = r'/home/zhuhao/DataSets/CatAndDog/test'
        model_dir = r'/home/zhuhao/DataSets/CatAndDog/model_file'
        logdir = r'/home/zhuhao/DataSets/CatAndDog/logs/train'

    if utils.isWinSys():
        train_dir = r'D:\DataSets\CatAndDog\train'
        test_dir = r'D:\DataSets\CatAndDog\test'
        model_dir = r'D:\DataSets\CatAndDog\model_file\alexnet'
        logdir = r'D:\DataSets\CatAndDog\logs\train'

    train_img_list, train_label_list = pre_pro.get_dogcat_img(train_dir)
    train_img_batch, train_label_batch = pre_pro.get_batch(train_img_list,train_label_list,227,227,batch_size=BATCH_SZIE,capacity=2000)
    train_label_batch = tf.one_hot(train_label_batch,depth=num_cls)

    test_img_list, test_label_list = pre_pro.get_dogcat_img(test_dir)
    test_img_batch, test_label_batch = pre_pro.get_batch(test_img_list,test_label_list,227,227,batch_size=BATCH_SZIE,capacity=2000)
    test_label_batch = tf.one_hot(test_label_batch,depth=num_cls)

    inputs = tf.placeholder(tf.float32,[None,227,227,3])
    labels = tf.placeholder(tf.float32,[None,num_cls])

    logits = AlexNet.AlexNetModel_slim(inputs=inputs,num_cls=num_cls,keep_prob=KEEP_PROB)

    train_loss = coms.loss(logits,labels)
    train_eval = coms.evaluation(logits,labels)
    train_optim = coms.optimizer(lr=5e-4,loss=train_loss,fun='mm')

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    max_acc = 0.

    with tf.Session() as sess:
        if utils.isHasGpu():
            dev = "/gpu:0"
        else:
            dev = "/cpu:0"
        with tf.device(dev):
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                if is_train:
                    if is_load_model:
                        ckpt = tf.train.get_checkpoint_state(model_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            saver.restore(sess,ckpt.model_checkpoint_path)
                            print('model load successful')
                        else:
                            print('model load falid')
                            return
                    n_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
                    logdir = os.path.join(logdir,n_time)
                    writer = tf.summary.FileWriter(logdir,sess.graph)
                    for epoch in range(EPOCH_NUM):
                        if coord.should_stop():
                            break
                        for step in range(ITER_NUM):
                            if coord.should_stop():
                                break
                            batch_train_im , batch_train_label = sess.run([train_img_batch,train_label_batch])
                            batch_test_im , batch_test_label = sess.run([test_img_batch,test_label_batch])

                            _, loss,w_summary = sess.run([train_optim,train_loss,summary_op],feed_dict={inputs:batch_train_im,labels:batch_train_label})

                            writer.add_summary(w_summary,(epoch * ITER_NUM + step))

                            print("epoch %d , step %d train end ,loss is : %f  ... ..." % (epoch, step, loss))

                            if epoch == 0 and step < (ITER_NUM / 3):
                                continue

                            if step % 200 == 0:
                                print('evaluation start ... ...')
                                ac_iter = int((len(test_img_list) / BATCH_SZIE))
                                ac_sum = 0.
                                for ac_count in range(ac_iter):
                                    accuracy = sess.run(train_eval,feed_dict={inputs:batch_test_im,labels:batch_test_label})
                                    ac_sum = ac_sum + accuracy
                                ac_mean = ac_sum / ac_iter
                                print('epoch %d , step %d , accuracy is %f'%(epoch,step,ac_mean))

                                if ac_mean >= max_acc:
                                    max_acc = ac_mean
                                    saver.save(sess, model_dir + '/' + 'dogcat' + str(epoch) + '_step' + str(step) + '.ckpt',
                                               global_step=step + 1)
                    print('saving last model ...')

                    saver.save(sess, model_dir + '/cifar10_last.ckpt')

                    print('train network task was run over')
                else:
                    pass
            except tf.errors.OutOfRangeError:
                print('done trainning --epoch files run out of')

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    main()