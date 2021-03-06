#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by tfrecords on 19-2-25
import tensorflow as tf
import numpy as np
import cv2
from imgaug import augmenters as iaa
import random
import coms.utils as utils
import os
from PIL import Image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfrecords(file,img_prob):
    file_queue = tf.train.string_input_producer([file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([],tf.int64),
                                           'image': tf.FixedLenFeature([],tf.string)
                                       })
    imgs = tf.decode_raw(features['image'],tf.uint8)
    imgs = tf.reshape(imgs,[img_prob[0],img_prob[1],img_prob[2]])   # [96,96,3]
    label = tf.cast(features['label'],tf.int64)


    # 图片标准化,等同于个通道减去rgb通道的均值
    imgs = tf.image.per_image_standardization(imgs)

    # imgs = tf.cast(imgs, tf.float32) / 255.  # 转换数据类型并归一化
    return imgs,label


def make_cifar10img_to_tfrecords(is_train,name):
    cls_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    imgs_dir = r''
    aim_dir = r''
    aim_name = r''

    if utils.isLinuxSys():
        if is_train:
            imgs_dir = r''
        else:
            imgs_dir = r''
        aim_dir = r''
    else:
        if is_train:
            imgs_dir = r'D:\DataSets\cifar\cifar\train'
        else:
            imgs_dir = r'D:\DataSets\cifar\cifar\test'

        aim_dir = r'D:\DataSets\cifar\cifar\tfrecords/'
    aim_name = aim_dir + name
    writer = tf.python_io.TFRecordWriter(aim_name + '.tfrecords')
    for file in os.listdir(imgs_dir):
        for index, cls_name in enumerate(cls_list):
            if cls_name in file:
                img_path = imgs_dir + '/' + file
                # img = load_img(img_path,(32,32))

                img = Image.open(img_path)
                img = img.resize((32, 32))
                img_raw = img.tobytes()

                feature = {
                    'label':_int64_feature(index),
                    'image':_bytes_feature(img_raw)
                }
                example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    writer.close()
    print('{}.tfrecords create over'.format(name))





if __name__ == '__main__':
    # make_stl10_tfrecord()
    make_cifar10img_to_tfrecords(True,'train')
    make_cifar10img_to_tfrecords(False,'test')

