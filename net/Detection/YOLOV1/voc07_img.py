#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by voc07_img on 19-4-28

import tensorflow as tf
import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from net.Detection.YOLOV1.config import COLORS,VOC07_CLASS
import cv2



# voc07_trainval_path = r'/home/zhuhao/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
# voc07_test_path = r'/home/zhuhao/DataSets/pascal_VOC/VOCtest_06-Nov2007/VOCdevkit/VOC2007'
# voc07_model = r'/home/zhuhao/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/model'

voc07_trainval_path = r'/home/ws/DataSets/pascal_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
voc07_test_path = r'/home/ws/DataSets/pascal_VOC/VOCtestval_06-Nov-2007/VOCdevkit/VOC2007'


class Pascal_voc(object):
    def __init__(self, rebuild=False):
        self.data_path = r''
        self.batch_size = 32
        self.image_size = 448
        self.cell_size = 7
        self.classes = VOC07_CLASS
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.phase = None
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None

    def prepare(self,phase):
        self.phase = phase
        gt_labels = self.load_labels()
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        if self.phase == 'train':
            self.data_path = voc07_trainval_path
            txtname = os.path.join(self.data_path,'ImageSets','Main','trainval.txt')
        elif self.phase == 'test':
            # 需要重新定义测试文件夹位置
            self.data_path = voc07_test_path
            txtname = os.path.join(voc07_test_path,'ImageSets','Main','test.txt')
        else:
            print('参数出错')
            return
        with open(txtname,'r') as f :
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:    # 没有对象，跳过
                continue
            imname = os.path.join(self.data_path,'JPEGImages', index + '.jpg')
            gt_labels.append({'imname':imname, 'label':label})
        return gt_labels


    def load_pascal_annotation(self,index):
        # imname = os.path.join(self.data_path,'JPEGImages', index + '.jpg')
        # im = cv2.imread(imname)
        # h_ratio = 1.0 * self.image_size / im.shape[0]
        # w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        size = tree.find('size')
        h_ratio = 1.0 * self.image_size / int(size.find('height').text)
        w_ratio = 1.0 * self.image_size / int(size.find('width').text)
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_idx[obj.find('name').text.lower().strip()]

            boxes = [(x2 + x1)/2.0, (y2+y1)/2.0, x2-x1, y2-y1]

            x_ind = int(boxes[0] * self.cell_size / self.image_size)    # 判断中点在第几个cell中
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1: #　如果第(y_ind,x_ind)格子中已经存在对象
                continue
            label[y_ind, x_ind, 0] = 1          # 对应(y_ind,x_ind)的cell中存在对象，置１
            label[y_ind, x_ind, 1:5] = boxes    # 储存x,y,w,h
            label[y_ind, x_ind, 5 + cls_ind] = 1    # 对应on-hot形式类设置为1

        return label, len(objs)

    def next_batch(self,gt_labels, batch_size):
        images = np.zeros((batch_size,self.image_size,self.image_size,3))
        labels = np.zeros((batch_size,self.cell_size,self.cell_size,25))
        count = 0

        while count < batch_size:
            imname = gt_labels[self.cursor]['imname']
            images[count,:,:,:] = self.image_read(imname)
            labels[count,:,:,:] = gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(gt_labels):
                np.random.shuffle(gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images,labels


    def image_read(self, imname):  # 读取图片
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))  # resize大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image = (image / 255.0) * 2.0 - 1.0
        image = image / 255.0
        return image


def test_voc07():
    pascal = Pascal_voc()

    gt_labels = pascal.prepare('test')


    for i in range(100):
        images,labels = pascal.next_batch(gt_labels,32)

        print(images.shape,labels.shape)
        return

    # check = 0
    # for gt_dict in gt_labels:
    #     img = cv2.imread(gt_dict['imname'])
    #     img = cv2.resize(img,(448,448))
    #     label = gt_dict['label']
    #     print('label shape',label.shape)
    #     for y_ind in range(7):
    #         for x_ind in range(7):
    #             if label[y_ind, x_ind, 0] == 1:
    #                 left_x ,left_y = label[y_ind,x_ind,1] - label[y_ind,x_ind,3]/2.0, label[y_ind,x_ind,2] - label[y_ind,x_ind,4]/2.0
    #                 right_x ,right_y = label[y_ind,x_ind,1] + label[y_ind,x_ind,3]/2.0, label[y_ind,x_ind,2] + label[y_ind,x_ind,4]/2.0
    #
    #                 left_x = int(left_x*448)
    #                 left_y = int(left_y*448)
    #                 right_x = int(right_x*448)
    #                 right_y = int(right_y*448)
    #
    #                 cls_text = VOC07_CLASS[np.argmax(label[y_ind,x_ind,5:])]
    #                 cv2.rectangle(img,(left_x,left_y),(right_x,right_y),COLORS[np.argmax(label[y_ind,x_ind,5:])])
    #                 cv2.putText(img,cls_text,(left_x,left_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    #
    #     cv2.imshow(str(check), img)
    #     check +=1
    #     if check >= 5:
    #         break
    # cv2.waitKey()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    test_voc07()
    #
    # print(dict(zip(VOC07_CLASS, range(len(VOC07_CLASS)))))