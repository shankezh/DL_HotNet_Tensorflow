#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by detector on 19-5-7

import net.Detection.YOLOV1.config as cfg
from coms.utils import isHasGpu
import tensorflow as tf
import cv2
import numpy as np
import colorsys
from net.Detection.YOLOV1.model import YOLO_Net

class Detector(object):
    def __init__(self,net):
        self.net = net
        self.image_size = cfg.DET_IMAGE_SIZE
        self.model_det_dir = cfg.DET_MODEL_DIR
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.PER_CELL_CHECK_BOXES
        self.classes = cfg.VOC07_CLASS
        self.num_class = len(self.classes)
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        if isHasGpu():
            gpu_option = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_option)
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)

    def draw_result(self, img, result):
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color, 1)
            cv2.putText(img, result[i][0], (x - w + 1, y - h + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ': %.2f%%' % (result[i][5] * 100))

    def detect(self,img):
        h,w,c = img.shape
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img = (img / 255.0) * 2.0 - 1.0
        img = img / 255.0
        imgs = np.reshape(img, (1,self.image_size,self.image_size,3))
        result = self.detect_from_cvmat(imgs)[0]
        for i in range(len(result)):
            result[i][1] *= (1.0 * w / self.image_size)
            result[i][2] *= (1.0 * h / self.image_size)
            result[i][3] *= (1.0 * w / self.image_size)
            result[i][4] *= (1.0 * h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs,self.net.is_training:False})
        print("net_shape:",net_output.shape)
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        print('rrr',results)
        return results

    def img_test(self):
        path = r'test/'
        self.load_model()
        for i in range(1,6):
            im_path = path + str(i) + '.jpg'
            img = cv2.imread(im_path)
            result = self.detect(img)
            print(result)
            self.draw_result(img, result)
            cv2.imshow(str(i), img)
        cv2.waitKey(0)

    def load_model(self):
        model_file = tf.train.latest_checkpoint(self.model_det_dir)
        self.saver.restore(self.sess, model_file)

    def interpret_output(self, output):

        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class)) # [7,7,2,20]
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))    # 类别 [0:7x7x20] - > [7,7,20]
        scales = np.reshape(output[self.boundary1:self.boundary2],(self.cell_size, self.cell_size, self.boxes_per_cell))  # [7x7x20:7x7x22] -> [7,7,2]  置信率
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))   # 坐标 [7x7x22:] -> [7,7,2,4]
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))  # [7,7,2]

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])


        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors

    def camera_detector(self, cap, wait=10):
        while (1):
            ret, frame = cap.read()
            result = self.detect(frame)

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    yolo = YOLO_Net(is_pre_training=False)
    # detector = Detector(yolo,0)
    detector = Detector(yolo)
    detector.img_test()
    # detector.image_detector(imname)
