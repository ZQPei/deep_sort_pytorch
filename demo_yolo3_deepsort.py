import os
import cv2
import numpy as np

from YOLO3 import YOLO3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time

class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("YOLO3/cfg/yolo_v3.cfg","YOLO3/yolov3.weights","YOLO3/cfg/coco.names", is_xywh=True)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        
    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        while self.vdo.grab(): 
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)]
            bbox_xywh, cls_conf, cls_ids = self.yolo3(im)
            if bbox_xywh is not None:
                mask = cls_ids==0
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3] *= 1.2
                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin,ymin))

            end = time.time()
            print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)
            


if __name__=="__main__":
    import sys
    if len(sys.argv) == 1:
        print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    else:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 800,600)
        det = Detector()
        det.open(sys.argv[1])
        det.detect()
