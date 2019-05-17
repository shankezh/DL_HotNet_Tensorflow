#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by config on 19-4-28
import tensorflow as tf

VOC07_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


COLORS = [[156,102,31], [255,127,80], [255,99,71], [255,255,0], [255,153,18],
          [227,207,87], [255,255,255], [202,235,216], [192,192,192], [251,255,242],
          [160,32,240], [218,112,214], [0,255,0], [255,0,0], [25,25,112],
          [3,168,158], [128,138,135], [128,118,105], [160,82,45], [8,46,84]]


# LOG_DIR = r'/home/ws/DataSets/cifar/logs/train/yolov1'
LOG_DIR = r'/home/ws/DataSets/pascal_VOC/VOC07/logs/yolov1'
# DET_MODEL_DIR = r'/home/ws/DataSets/pascal_VOC/VOC07/model_file/yolov1/det' # 加载识别训练模型需要改对应路径
DET_MODEL_DIR = r'/home/zhuhao/DataSets/pascal_VOC/VOC07/model_file/yolov1/det'
CLS_MODEL_DIR = r'/home/ws/DataSets/cifar/model_file/yolov1/cls'    # 加载预训练模型需要改对应路径


EPOCH = 150
ITER_STEP = 156    # cifar-10: 50000 / 32 = 1560       # voc07 train_val 5000 / 32 = 156

THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
PRE_TRAIN_NUM = 10

PER_CELL_CHECK_BOXES = 2

CELL_SIZE = 7
DET_IMAGE_SIZE = 448

OBJ_CONFIDENCE_SCALE = 1.0
NO_OBJ_CONFIDENCE_SCALE = 0.5
COORD_SCALE = 5.0
CLASS_SCALE = 1.0


BATCH_SIZE = 128
KEEP_PROB = 0.5


BATCH_NORM_PARAMS = {
        'decay': 0.998,
        'epsilon': 0.001,
        'scale': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

if __name__ == '__main__':
   pass
