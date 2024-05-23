import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from detector.YOLOv5.utils.augmentations import letterbox
from detector.YOLOv5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from detector.YOLOv5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from detector.YOLOv5.utils.plots import Annotator, colors, save_one_box
from detector.YOLOv5.utils.torch_utils import select_device, time_sync


class YOLOv5(object):
    def __init__(self, weight='yolov5s.pt', data='data/coco128.yaml', imgsz=[640, 640],
                 conf_thres=0.25, nms_thres=0.45, max_det=1000, device='cuda:0', dnn=False):
        super().__init__()
        self.device = select_device(device)
        self.net = DetectMultiBackend(weight, device=self.device, dnn=dnn, data=data)
        self.stride, self.class_names, self.pt = self.net.stride, self.net.names, self.net.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_det = max_det

    def __call__(self, im0, augment=False, save_result=False):
        # im shape is [H, W, 3] and RGB
        # read image
        bs = 1
        img = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        # preprocess image
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]

        # model inference
        # self.net.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        pred = self.net(img, augment=augment)
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres,
                                   classes=None, agnostic=False, max_det=self.max_det)[0]

        # postprocess det
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

        if save_result is True:
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            for *xyxy, conf, cls in pred:
                c = int(cls)  # integer class
                label = '{:} {:.2f}'.format(self.names[c], conf)
                annotator.box_label(xyxy, label, color=colors(c, False))
            im0 = annotator.result()

        pred[:, :4] = xyxy2xywh(pred[:, :4])
        xywh = pred[:, :4].cpu().numpy()
        conf = pred[:, 4].cpu().numpy()
        cls = pred[:, 5].cpu().numpy()
        return (xywh, conf, cls) if not save_result else (xywh, conf, cls, im0)


def demo():
    yolo = YOLOv5(weight='yolov5s.pt', data='data/coco128.yaml')
    root = "./data/images"
    files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids, img_ = yolo(img, save_result=True)
        # imshow
        cv2.namedWindow("yolo")
        cv2.imshow("yolo", img_[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
