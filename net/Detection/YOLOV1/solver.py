#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by solver on 19-5-6
import tensorflow as tf
from net.Detection.YOLOV1.model import YOLO_Net
import net.Detection.YOLOV1.config as cfg
import tensorflow.contrib.slim as slim
from net.Detection.YOLOV1.voc07_img import Pascal_voc
from coms.learning_rate import CLR_EXP_RANGE
from coms.utils import  isHasGpu,isLinuxSys
import time,os
from coms.pre_process import get_cifar10_batch
import net.Detection.YOLOV1.voc07_tfrecord as VOC07RECORDS

class Solver(object):
    def __init__(self,net,data,tf_records=False):
        self.net = net
        self.data = data
        self.tf_records = tf_records
        self.batch_size = cfg.BATCH_SIZE
        self.clr = CLR_EXP_RANGE()
        self.log_dir = cfg.LOG_DIR
        self.model_cls_dir = cfg.CLS_MODEL_DIR
        self.model_det_dir = cfg.DET_MODEL_DIR
        self.learning_rate = tf.placeholder(tf.float32)
        self.re_train = True
        tf.summary.scalar('learning_rate',self.learning_rate)
        self.optimizer = self.optimizer_bn(lr=self.learning_rate,loss=self.net.total_loss)
        if isHasGpu():
            gpu_option = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_option)
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.summary_op = tf.summary.merge_all()
        n_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, n_time),self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=4)


    def train_classify(self):
        self.set_classify_params()
        max_acc = 0.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        for epoch in range(cfg.EPOCH):
            for step in range(1,cfg.ITER_STEP+1):
                learning_rate_val = self.clr.calc_lr(step,cfg.ITER_STEP+1,0.001,0.01,gamma=0.9998)
                train_img_batch, train_label_batch = self.sess.run([self.train_img_batch,self.train_label_batch])
                feed_dict_train = {self.net.images:train_img_batch, self.net.labels:train_label_batch, self.net.is_training:True,self.learning_rate:learning_rate_val}
                _, summary_op, batch_train_loss, batch_train_acc = self.sess.run([self.optimizer, self.summary_op,self.net.total_loss,self.net.evalution],feed_dict=feed_dict_train)

                global_step = int(epoch * cfg.ITER_STEP + step + 1)
                print("epoch %d , step %d train end ,loss is : %f ,accuracy is %f ... ..." % (epoch, step, batch_train_loss, batch_train_acc))
                train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag='train_loss', simple_value=batch_train_loss)
                        , tf.Summary.Value(tag='train_batch_accuracy', simple_value=batch_train_acc)
                        , tf.Summary.Value(tag='learning_rate', simple_value=learning_rate_val)])
                self.writer.add_summary(train_summary,global_step=global_step)
                self.writer.add_summary(summary_op,global_step=global_step)
                self.writer.flush()


                if step % 100 == 0:
                    print('test sets evaluation start ...')
                    ac_iter = int(10000 / self.batch_size)  # cifar-10测试集数量10000张
                    ac_sum = 0.
                    loss_sum = 0.
                    for ac_count in range(ac_iter):
                        batch_test_img, batch_test_label = self.sess.run([self.test_img_batch, self.test_label_batch])
                        feed_dict_test = {self.net.images: batch_test_img,self.net.labels: batch_test_label,self.net.is_training: False,self.learning_rate:learning_rate_val}
                        test_loss, test_accuracy = self.sess.run([self.net.total_loss, self.net.evalution],feed_dict=feed_dict_test)

                        ac_sum += test_accuracy
                        loss_sum += test_loss
                    ac_mean = ac_sum / ac_iter
                    loss_mean = loss_sum / ac_iter
                    print('epoch {} , step {} , accuracy is {}'.format(str(epoch), str(step), str(ac_mean)))
                    test_summary = tf.Summary(
                        value=[tf.Summary.Value(tag='test_loss', simple_value=loss_mean)
                            , tf.Summary.Value(tag='test_accuracy', simple_value=ac_mean)])
                    self.writer.add_summary(test_summary, global_step=global_step)
                    self.writer.flush()

                    if ac_mean >= max_acc:
                        max_acc = ac_mean
                        self.saver.save(self.sess, self.model_cls_dir + '/' + 'cifar10_{}_step_{}.ckpt'.format(str(epoch),str(step)), global_step=step)
                        print('max accuracy has reaching ,save model successful ...')
        print('train network task was run over')


    def set_classify_params(self):
        self.train_img_batch,self.train_label_batch = get_cifar10_batch(is_train=True,batch_size=self.batch_size,num_cls=cfg.PRE_TRAIN_NUM,img_prob=[224,224,3])
        self.test_img_batch,self.test_label_batch = get_cifar10_batch(is_train=False,batch_size=self.batch_size,num_cls=cfg.PRE_TRAIN_NUM,img_prob=[224,224,3])

    def train_detector(self):
        self.set_detector_params()
        for epoch in range(cfg.EPOCH):
            for step in range(1,cfg.ITER_STEP+1):
                global_step = int(epoch * cfg.ITER_STEP + step + 1)
                learning_rate_val = self.clr.calc_lr(step,cfg.ITER_STEP+1,0.0001,0.0005,gamma=0.9998)
                if self.tf_records:
                    train_images, train_labels = self.sess.run(self.train_next_elements)
                else:
                    train_images, train_labels = self.data.next_batch(self.gt_labels_train, self.batch_size)
                feed_dict_train = {self.net.images:train_images,self.net.labels:train_labels,self.learning_rate:learning_rate_val,self.net.is_training:True}
                _,summary_str,train_loss = self.sess.run([self.optimizer,self.summary_op,self.net.total_loss],feed_dict=feed_dict_train)
                print("epoch %d , step %d train end ,loss is : %f  ... ..." % (epoch, step, train_loss))
                self.writer.add_summary(summary_str,global_step)

                if step % 50 ==0:
                    print('test sets start ...')
                    # test sets sum :4962
                    sum_loss = 0.
                    # test_iter = int (4962 / self.batch_size)
                    test_iter = 10  # 取10个批次求均值
                    for _ in range(test_iter):
                        if self.tf_records:
                            test_images, test_labels = self.sess.run(self.test_next_elements)
                        else:
                            test_images,test_labels = self.data.next_batch(self.gt_labels_test,self.batch_size)
                        feed_dict_test = {self.net.images:test_images,self.net.labels:test_labels,self.net.is_training:False}
                        loss_iter = self.sess.run(self.net.total_loss,feed_dict=feed_dict_test)
                        sum_loss += loss_iter

                    mean_loss = sum_loss/test_iter
                    print('epoch {} , step {} , test loss is {}'.format(str(epoch), str(step), str(mean_loss)))
                    test_summary = tf.Summary(
                        value=[tf.Summary.Value(tag='test_loss', simple_value=mean_loss)])
                    self.writer.add_summary(test_summary, global_step=global_step)
                    self.writer.flush()

            self.saver.save(self.sess,self.model_det_dir+'/' + 'det_voc07_{}_step_{}.ckpt'.format(str(epoch),str(step)), global_step=step)
            print('save model successful ...')

    def set_detector_params(self):
        if self.tf_records:
            train_records_path = r'/home/ws/DataSets/pascal_VOC/VOC07/tfrecords' + '/trainval.tfrecords'
            test_records_path = r'/home/ws/DataSets/pascal_VOC/VOC07/tfrecords' + '/test.tfrecords'
            train_datasets = VOC07RECORDS.DataSets(record_path=train_records_path,batch_size=self.batch_size)
            train_gen = train_datasets.transform(shuffle=True)
            train_iterator = train_gen.make_one_shot_iterator()
            self.train_next_elements = train_iterator.get_next()
            test_datasets = VOC07RECORDS.DataSets(record_path=test_records_path, batch_size=self.batch_size)
            test_gen = test_datasets.transform(shuffle=True)
            test_iterator = test_gen.make_one_shot_iterator()
            self.test_next_elements = test_iterator.get_next()
        else:
            self.gt_labels_train = self.data.prepare('train')
            self.gt_labels_test = self.data.prepare('test')
        if self.re_train:
            self.load_det_model()
        else:
            self.load_pre_train_model()


    def load_pre_train_model(self):
        net_vars = slim.get_model_variables()
        model_file = tf.train.latest_checkpoint(self.model_cls_dir)
        reader = tf.train.NewCheckpointReader(model_file)
        model_vars = reader.get_variable_to_shape_map()
        exclude = ['yolov1/classify_fc1/weights', 'yolov1/classify_fc1/biases']

        vars_restore_map = {}
        for var in net_vars:
            if var.op.name in model_vars and var.op.name not in exclude:
                vars_restore_map[var.op.name] = var

        self.saver = tf.train.Saver(vars_restore_map,max_to_keep=4)
        self.saver.restore(self.sess, model_file)
        self.saver = tf.train.Saver(var_list=net_vars,max_to_keep=4)

        print('load pre-train model successful ...')

    def load_det_model(self):
        # self.saver = tf.train.Saver(max_to_keep=4)
        net_vars = slim.get_model_variables()
        self.saver = tf.train.Saver(net_vars,max_to_keep=4)

        model_file = tf.train.latest_checkpoint(self.model_det_dir)
        self.saver.restore(self.sess, model_file)
        print('load model successful ...')



    # 带BN的训练函数
    def optimizer_bn(self,lr, loss, mom=0.9, fun='mm'):
        with tf.name_scope('optimzer_bn'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
                train_op = slim.learning.create_train_op(loss, optim)
        return train_op



def train_classify():
    yolov1 = YOLO_Net(is_pre_training=True)

    sovler = Solver(net= yolov1,data=0)
    print('start ...')
    sovler.train_classify()

def train_detector():
    yolov1 = YOLO_Net(is_pre_training=False)
    pasvoc07 = Pascal_voc()
    sovler = Solver(net=yolov1,data=pasvoc07)
    print('start train ...')
    sovler.train_detector()

def train_detector_with_records():
    yolov1 = YOLO_Net(is_pre_training=False)
    sovler = Solver(net=yolov1,data=0,tf_records=True)
    print('start train ...')
    sovler.train_detector()

if __name__ == '__main__':
    train_detector_with_records()