#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by voc07_tfrecord on 19-4-25
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
from net.Detection.YOLOV1.config import COLORS,VOC07_CLASS
import net.Detection.YOLOV1.config as cfg
import cv2
from tqdm import tqdm


voc_tf_07_save_path = r'/home/zhuhao/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/tfrecords'
# voc_tf_07_save_path = r'/home/ws/DataSets/pascal_VOC/VOC07/tfrecords'

voc07_path = r'/home/zhuhao/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'

voc07_train = r'/home/ws/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
voc07_test = r'/home/ws/DataSets/pascal_VOC/VOCtestval_06-Nov-2007/VOCdevkit/VOC2007'



class To_tfrecords(object):
    def __init__(self,
                 load_folder=voc07_test,
                 txt_file='trainval.txt',
                 # txt_file = 'test.txt',
                 save_folder=voc_tf_07_save_path):
        self.load_folder = load_folder
        self.save_folder = save_folder
        self.txt_file = txt_file
        self.usage = self.txt_file.split('.')[0]
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.classes = VOC07_CLASS
        self.cell_size = cfg.CELL_SIZE
        self.image_size = cfg.DET_IMAGE_SIZE
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))


    def load_and_resize(self,path,size):
        img = Image.open(path)
        img = img.resize((size, size))
        img_raw = img.tobytes()
        return img_raw

    def transform(self):
        # 1. 获取作为训练集/验证集的图片编号
        txt_file = os.path.join(self.load_folder, 'ImageSets', 'Main', self.txt_file)
        with open(txt_file) as f:
            image_index = [_index.strip() for _index in f.readlines()]

        # 2. 开始循环写入每一张图片以及标签到tfrecord文件
        with tf.python_io.TFRecordWriter(os.path.join(
                self.save_folder, self.usage + '.tfrecords')) as writer:
            for _index in tqdm(image_index, desc='开始写入tfrecords数据'):
                filename = os.path.join(self.load_folder, 'JPEGImages', _index) + '.jpg'
                xml_file = os.path.join(self.load_folder, 'Annotations', _index) + '.xml'
                assert os.path.exists(filename)
                assert os.path.exists(xml_file)

                img = tf.gfile.FastGFile(filename, 'rb').read()
                # img = self.load_and_resize(filename,self.image_size)

                # 解析label文件
                label,num = self._parser_xml(xml_file)

                # filename = filename.encode()
                # 需要将其转换一下用str >>> bytes encode()
                if num == 0:
                    continue
                label = [float(_) for _ in label]
                # Example协议
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                    # 'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
                    }))
                writer.write(example.SerializeToString())
            writer.close()

    def _parser_xml(self, xml_file):

        tree = ET.parse(xml_file)
        # 得到某个xml_file文件中所有的object
        objs = tree.findall('object')
        size = tree.find('size')
        h_ratio = 1.0 * self.image_size / int(size.find('height').text)
        w_ratio = 1.0 * self.image_size / int(size.find('width').text)

        labels = []
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_idx[obj.find('name').text.lower().strip()]

            labels.extend([x1,y1,x2,y2,cls_ind])

        return labels,len(objs)



class DataSets(object):
    def __init__(self,record_path,batch_size):
        self.record_path = record_path
        self.batch_size = batch_size
        self.image_size = cfg.DET_IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.VOC07_CLASS

    def transform(self,shuffle):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(DataSets._parser)
        dataset = dataset.map(map_func=lambda image,label:tf.py_func(func=self._process,inp=[image,label],Tout=[tf.float32,tf.float32]),num_parallel_calls=8)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset

    def _process(self,image,label):
        label = np.reshape(label,(-1,5))
        label_list = [list(label[row, :]) for row in range(label.shape[0])]
        yolo_label = self._to_yolo(label_list)
        return image,yolo_label


    def _to_yolo(self,label_list):
        yolo_label = np.zeros((self.cell_size, self.cell_size, 5+len(self.classes)),dtype=np.float32)

        for label in label_list:
            x1, y1, x2, y2, cls_ind = label

            boxes = [(x2+x1)/2.0, (y2+y1)/2.0, x2-x1, y2-y1]
            x_ind = int(boxes[0]*self.cell_size/self.image_size)    # 判断横向在第几个cell
            y_ind = int(boxes[1]*self.cell_size/self.image_size)    # 判断纵向在第几个cell
            if yolo_label[y_ind,x_ind,0] == 1:
                continue
            yolo_label[y_ind,x_ind,0] = 1 # 对应cell存在对象，置位1
            yolo_label[y_ind,x_ind,1:5] = boxes
            yolo_label[y_ind,x_ind,5+ int(cls_ind)] = 1   # 对应类别的位置设置为1
        return yolo_label


    @staticmethod
    def _parser(record):
        features = {"img": tf.FixedLenFeature((), tf.string),
                    "label": tf.VarLenFeature(tf.float32)}
        features = tf.parse_single_example(record, features)
        img = tf.image.decode_jpeg(features["img"])
        img = tf.image.resize_images(img,[448,448])
        img = tf.cast(img,tf.float32) /255.

        # img = tf.decode_raw(features['img'],tf.float32)
        # img = tf.reshape(img,[448,448,3])
        # img = img / 255.
        label = features["label"].values
        return img, label




def showTest(img,label):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(7):
        for j in range(7):
            if label[i,j,0] == 0:
                continue
            xc,yc,w,h = label[i,j,1:5]
            lx = (xc - 0.5 * w ) * 448
            ly = (yc - 0.5 * h) * 448

            rx = (xc + 0.5 * w )* 448
            ry = (yc + 0.5 * h )* 448
            img = cv2.rectangle(img,(int(lx),int(ly)),(int(rx),int(ry)),(255,0,0))
            print(lx,ly,rx,ry)
    cv2.imshow('a',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    train_generator = DataSets(record_path=voc_tf_07_save_path+'/trainval.tfrecords',batch_size=cfg.BATCH_SIZE)
    train_dataset = train_generator.transform(shuffle=True)
    iterator = train_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cou= 0
        # while(True):

        images,labels = sess.run(next_element)
        images *= 255.
        # cou +=1
        # print(cou,images.shape,labels.shape)
        img = sess.run(tf.cast(images[0],tf.uint8))
        draw(img,labels[0])

def draw(img,label):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(7):
        for j in range(7):
            if label[i,j,0] == 0:
                continue
            print('find')
            xc, yc, w, h = label[i, j, 1:5]
            lx = xc - w/2.
            ly = yc - h/2.
            rx = xc + w/2.
            ry = yc + h/2.
            img = cv2.rectangle(img,(int(lx),int(ly)),(int(rx),int(ry)),(255,0,0))
            # print(label[i,j,5:])
            img = cv2.putText(img,VOC07_CLASS[np.argmax(label[i,j,5:])],(int(lx),int(ly)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(255,255,255))
    cv2.imshow('a', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # to_recotds = To_tfrecords()
    # to_recotds.transform()
    #
    test()
    # train_generator = Dataset(filenames=voc_tf_07_save_path + '/trainval.tfrecords')
    # train_dataset = train_generator.transform()
    # iterator = train_dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # # 检查生成的图像及 bounding box
    #
    # count = 0
    # check = 2
    # with tf.Session() as sess:
    #     for i in range(10):
    #         images, labels = sess.run(next_element)
    #         print(labels.shape)
    #         while count < check:
    #             image, label = images[count, ...], labels[count, ...]
    #             showTest(image,label)
    #             count += 1