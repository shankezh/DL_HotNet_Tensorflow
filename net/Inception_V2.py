#!/usr/bin/env python3.6.3
# encoding: utf-8
# @Time    : 2019/3/2 14:28
# @Author  : Echo
# set(  = "https://blog.csdn.net/shankezh" )
# @contact: cloud_happy@163.com
# @Site    : 
# @File    : VGG.py
# @Software: PyCharm

import tensorflow as tf
import coms.utils as utils
import coms.pre_process as pre_pro
import coms.coms as coms
import net.GoogLeNet.InceptionV2 as InceptionV2
import coms.learning_rate as LR_Tools
import time
import cv2
import numpy as np
import os


def run():
    model_dir = ''
    logdir = ''
    img_prob = [224, 224, 3]
    num_cls = 10
    is_train = False
    is_load_model = False
    is_stop_test_eval = True
    BATCH_SIZE = 100
    EPOCH_NUM = 150
    ITER_NUM = 500  # 50000 / 100
    LEARNING_RATE_VAL = 0.001

    if utils.isLinuxSys():
        logdir = r''
        model_dir = r''
    else:
        model_dir = r'D:\DataSets\cifar\cifar\model_flie\inceptionv2'
        logdir = r'D:\DataSets\cifar\cifar\logs\train\inceptionv2'

    if is_train:
        train_img_batch, train_label_batch = pre_pro.get_cifar10_batch(is_train = True, batch_size=BATCH_SIZE, num_cls=num_cls,img_prob=[224,224,3])
        test_img_batch, test_label_batch = pre_pro.get_cifar10_batch(is_train=False,batch_size=BATCH_SIZE,num_cls=num_cls,img_prob=[224,224,3])

    inputs = tf.placeholder(tf.float32,[None, img_prob[0], img_prob[1], img_prob[2]])
    labels = tf.placeholder(tf.float32,[None, num_cls])
    is_training = tf.placeholder(tf.bool)
    LEARNING_RATE = tf.placeholder(tf.float32)

    calc_lr = LR_Tools.CLR_EXP_RANGE()

    # layer_batch_norm_params = {
    #     'training': is_training
    # }

    logits = InceptionV2.V2_slim(inputs, num_cls, is_training=is_training)

    train_loss = coms.loss(logits,labels)
    train_optim = coms.optimizer_bn(lr=LEARNING_RATE,loss=train_loss)
    train_eval = coms.evaluation(logits,labels)


    saver = tf.train.Saver(max_to_keep=4)
    max_acc = 0.

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if utils.isHasGpu():
            dev = '/gpu:0'
        else:
            dev = '/cpu:0'
        with tf.device(dev):
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess= sess, coord=coord)

            try:
                if is_train:
                    if is_load_model:
                        ckpt = tf.train.get_checkpoint_state(model_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            saver.restore(sess,ckpt.model_checkpoint_path)
                            print('model load successful ...')
                        else:
                            print('model load failed ...')
                            return
                    n_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
                    logdir = os.path.join(logdir, n_time)
                    writer = tf.summary.FileWriter(logdir, sess.graph)

                    for epoch in range(EPOCH_NUM):
                        if coord.should_stop():
                            print('coord should stop ...')
                            break
                        for step in range(1,ITER_NUM+1):
                            if coord.should_stop():
                                print('coord should stop ...')
                                break

                            script_kv = utils.readFile('train_script')
                            if script_kv != None:
                                if 'iter' in script_kv.keys():
                                    # 找到对应step，准备保存模型参数，或者修改学习速率
                                    if int(script_kv['iter']) == step:
                                        if 'lr' in script_kv.keys():
                                            LEARNING_RATE_VAL = float(script_kv['lr'])
                                            print('read train_scrpit file and update lr to {}'.format(LEARNING_RATE_VAL))
                                        if 'save' in script_kv.keys():
                                            saver.save(sess,model_dir + '/' + 'cifar10_{}_step_{}.ckpt'.format(str(epoch),str(step)),global_step=step)
                                            print('read train_script file and save model successful ...')
                            LEARNING_RATE_VAL = calc_lr.calc_lr(step,ITER_NUM,0.001,0.01,gamma=0.9998)
                            # LEARNING_RATE_VAL = coms.clr(step,2*ITER_NUM,0.001,0.006)

                            batch_train_img, batch_train_label = sess.run([train_img_batch,train_label_batch])

                            _, batch_train_loss, batch_train_acc = sess.run([train_optim,train_loss,train_eval],feed_dict={inputs:batch_train_img,
                                                                                                                           labels:batch_train_label,
                                                                                                                           LEARNING_RATE:LEARNING_RATE_VAL,
                                                                                                                           is_training:is_train})
                            global_step = int(epoch * ITER_NUM + step + 1)

                            print("epoch %d , step %d train end ,loss is : %f ,accuracy is %f ... ..." % (epoch, step, batch_train_loss, batch_train_acc))

                            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss',simple_value=batch_train_loss)
                                                              ,tf.Summary.Value(tag='train_batch_accuracy',simple_value=batch_train_acc)
                                                              ,tf.Summary.Value(tag='learning_rate',simple_value=LEARNING_RATE_VAL)])

                            writer.add_summary(train_summary,global_step)


                            writer.flush()

                            if is_stop_test_eval:
                                if not is_load_model:
                                    if epoch < 3:
                                        continue


                            if step % 100 == 0:
                                print('test sets evaluation start ...')
                                ac_iter = int(10000/BATCH_SIZE) # cifar-10测试集数量10000张
                                ac_sum = 0.
                                loss_sum = 0.
                                for ac_count in range(ac_iter):
                                    batch_test_img, batch_test_label = sess.run([test_img_batch,test_label_batch])
                                    test_loss, test_accuracy = sess.run([train_loss,train_eval],feed_dict={inputs:batch_test_img,
                                                                                                           labels:batch_test_label,
                                                                                                           is_training:False})
                                    ac_sum += test_accuracy
                                    loss_sum += test_loss
                                ac_mean = ac_sum / ac_iter
                                loss_mean = loss_sum / ac_iter
                                print('epoch {} , step {} , accuracy is {}'.format(str(epoch),str(step),str(ac_mean)))
                                test_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag='test_loss', simple_value=loss_mean)
                                        , tf.Summary.Value(tag='test_accuracy', simple_value=ac_mean)])
                                writer.add_summary(test_summary,global_step=global_step)
                                writer.flush()

                                if ac_mean >= max_acc:
                                    max_acc = ac_mean
                                    saver.save(sess, model_dir + '/' + 'cifar10_{}_step_{}.ckpt'.format(str(epoch),str(step)),global_step=step)
                                    print('max accuracy has reaching ,save model successful ...')
                    # print('saving last model ...')
                    # saver.save(sess, model_dir + '/' + 'cifar10_last.ckpt')
                    print('train network task was run over')
                else:
                    model_file = tf.train.latest_checkpoint(model_dir)
                    saver.restore(sess, model_file)
                    cls_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                'truck']
                    for i in range(1, 11):
                        name = str(i) + '.jpg'
                        img = cv2.imread(name)
                        img = cv2.resize(img, (32, 32))
                        img = cv2.resize(img, (224, 224))
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255.
                        img = np.array([img])
                        res = sess.run(logits, feed_dict={inputs:img , is_training:False})

                        # print(res)
                        print('{}.jpg detect result is : '.format(str(i)) + cls_list[np.argmax(res)] )


                # img = cv2.imread("16_dog.png")
                # img = cv2.resize(img, (224, 224))
                # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = img / 255.
                # img = np.array([img])
                # res = sess.run(logits, feed_dict={inputs: img, is_training: False})
                # print(res)
                # print('{}.jpg detect result is : '.format(str('16_dog')) + cls_list[np.argmax(res)])
                #
                # img = cv2.imread("5_frog.png")
                # img = cv2.resize(img, (224, 224))
                # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = img / 255.
                # img = np.array([img])
                # res = sess.run(logits, feed_dict={inputs: img, is_training: False})
                # print(res)
                # print('{}.jpg detect result is : '.format(str('5_frog')) + cls_list[np.argmax(res)])
                #
                # img = cv2.imread("11_truck.png")
                # img = cv2.resize(img, (224, 224))
                # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = img / 255.
                # img = np.array([img])
                # res = sess.run(logits, feed_dict={inputs: img, is_training: False})
                # print(res)
                # print('{}.jpg detect result is : '.format(str('11_truck')) + cls_list[np.argmax(res)])
            except tf.errors.OutOfRangeError:
                print('done training -- opoch files run out of ...')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    run()