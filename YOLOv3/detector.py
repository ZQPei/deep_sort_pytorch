import torch
# import time
import numpy as np
import cv2

from darknet import Darknet
from yolo_utils import get_all_boxes, nms, plot_boxes_cv2


class YOLOv3(object):
    def __init__(self, cfgfile, weightfile, namesfile, use_cuda=True, is_plot=False, is_xywh=False, conf_thresh=0.3, nms_thresh=0.4):
        # net definition
        self.net = Darknet(cfgfile)
        self.net.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.net.width, self.net.height
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_plot = is_plot
        self.is_xywh = is_xywh
        self.class_names = self.load_class_names(namesfile)

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img = ori_img.astype(np.float)/255.

        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)
        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)
            boxes = get_all_boxes(out_boxes, self.conf_thresh, self.net.num_classes, use_cuda=self.use_cuda)[0]
            boxes = nms(boxes, self.nms_thresh)
            # print(boxes)
        # plot boxes
        if self.is_plot:
            return self.plot_bbox(ori_img, boxes)
        if len(boxes)==0:
            return None,None,None
        
        height , width = ori_img.shape[:2]
        boxes = np.vstack(boxes)
        bbox = np.empty_like(boxes[:,:4])
        if self.is_xywh:
            # bbox x y w h
            bbox[:,0] = boxes[:,0]*width
            bbox[:,1] = boxes[:,1]*height
            bbox[:,2] = boxes[:,2]*width
            bbox[:,3] = boxes[:,3]*height
        else:
            # bbox xmin ymin xmax ymax
            bbox[:,0] = (boxes[:,0]-boxes[:,2]/2.0)*width
            bbox[:,1] = (boxes[:,1]-boxes[:,3]/2.0)*height
            bbox[:,2] = (boxes[:,0]+boxes[:,2]/2.0)*width
            bbox[:,3] = (boxes[:,1]+boxes[:,3]/2.0)*height
        cls_conf = boxes[:,5]
        cls_ids = boxes[:,6]
        return bbox, cls_conf, cls_ids

    def load_class_names(self,namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def plot_bbox(self, ori_img, boxes):
        img = ori_img
        height , width = img.shape[:2]
        for box in boxes:
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2]/2.0) * width).item()))
            y1 = int(round(((box[1] - box[3]/2.0) * height).item()))
            x2 = int(round(((box[0] + box[2]/2.0) * width).item()))
            y2 = int(round(((box[1] + box[3]/2.0) * height).item()))
            cls_conf = box[5]
            cls_id = box[6]
            # import random
            # color = random.choices(range(256),k=3)
            color = [int(x) for x in np.random.randint(256,size=3)]
            # put texts and rectangles
            img = cv2.putText(img, self.class_names[cls_id], (x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            img = cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        return img

if __name__ == '__main__':
    yolo3 = YOLOv3("cfg/yolo_v3.cfg","yolov3.weights","cfg/coco.names", is_plot=True)
    print("yolo3.size =",yolo3.size)
    import os
    root = "../demo"
    files = [os.path.join(root,file) for file in os.listdir(root)]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo3(img)
        # save results
        # cv2.imwrite("../result/{}".format(os.path.basename(filename)),res[:,:,(2,1,0)])
        # imshow
        # cv2.namedWindow("yolo3", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("yolo3", 600,600)
        # cv2.imshow("yolo3",res[:,:,(2,1,0)])
        # cv2.waitKey(0)

