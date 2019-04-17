import torch

# import time
import numpy as np
import cv2
from darknet import Darknet

from yolo_utils import get_all_boxes, nms, plot_boxes_cv2


class YOLO3(object):
    def __init__(
        self,
        cfgfile,
        weightfile,
        namesfile,
        use_cuda=True,
        is_plot=False,
        is_xywh=False,
        half=False,
    ):
        # net definition
        self.half = half
        self.net = Darknet(cfgfile)
        self.net.load_weights(weightfile)
        print("Loading weights from %s... Done!" % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)
        if half:
            self.net = self.net.half()

        # constants
        self.size = self.net.width, self.net.height
        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        self.use_cuda = use_cuda
        self.is_plot = is_plot
        self.is_xywh = is_xywh
        self.class_names = self.load_class_names(namesfile)

    def __call__(self, imgs, ori_ims):
        # img to tensor
        # forward
        with torch.no_grad():
            imgs = imgs.to(self.device)
            batch_out_boxes = self.net(imgs)
            batch_boxes = get_all_boxes(
                batch_out_boxes, self.conf_thresh, self.net.num_classes, self.use_cuda
            )
            batch_boxes = list(
                map(lambda boxes: nms(boxes, self.nms_thresh), batch_boxes)
            )
        # plot boxes
        if self.is_plot:
            return [
                self.plot_bbox(ori_ims[i], batch_boxes[i]) for i in range(len(ori_ims))
            ]

        height, width = ori_ims[-1].shape[:2]
        batch_bbox, batch_cls_conf, batch_cls_ids = [], [], []
        for boxes in batch_boxes:
            if len(boxes) == 0:
                batch_bbox.append(None)
                batch_cls_conf.append(None)
                batch_cls_ids.append(None)
                continue

            boxes = np.vstack(boxes)
            bbox = np.empty_like(boxes[:, :4])
            if self.is_xywh:
                # bbox x y w h
                bbox[:, 0] = boxes[:, 0] * width
                bbox[:, 1] = boxes[:, 1] * height
                bbox[:, 2] = boxes[:, 2] * width
                bbox[:, 3] = boxes[:, 3] * height
            else:
                # bbox xmin ymin xmax ymax
                bbox[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
                bbox[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
                bbox[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
                bbox[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
            batch_bbox.append(bbox)
            batch_cls_conf.append(boxes[:, 5])
            batch_cls_ids.append(boxes[:, 6])
        return batch_bbox, batch_cls_conf, batch_cls_ids

    def load_class_names(self, namesfile):
        with open(namesfile, "r", encoding="utf8") as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def plot_bbox(self, ori_img, boxes):
        img = ori_img
        height, width = img.shape[:2]
        for box in boxes:
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
            y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
            x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
            y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
            cls_conf = box[5]
            cls_id = box[6]
            # import random
            # color = random.choices(range(256),k=3)
            color = [int(x) for x in np.random.randint(256, size=3)]
            # put texts and rectangles
            img = cv2.putText(
                img,
                self.class_names[cls_id],
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    def load_class_names(self, namesfile):
        with open(namesfile, "r", encoding="utf8") as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def run(self):
        while 1:
            while len(self.postprocessing_queue) == 0:
                time.sleep(0.2)
            (
                batch_boxes,
                ims,
                ori_ims,
                area,
                self.im_width,
                self.im_height,
            ) = postprocessing_queue.pop(0)
            batch_bbox_xywh, batch_cls_conf, batch_cls_ids = self.postprocess(
                batch_boxes, ori_ims
            )

            xmin, ymin, xmax, ymax = area
            if self.write_video and self.output is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(
                    "demo.avi", fourcc, 2.5, (self.im_width, self.im_height)
                )

            assert len(batch_bbox_xywh) == self.batch_size
            for i, (bbox_xywh, cls_conf, cls_ids) in enumerate(
                zip(batch_bbox_xywh, batch_cls_conf, batch_cls_ids)
            ):
                if bbox_xywh is not None:
                    mask = cls_ids == 0
                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:, 3] *= 1.2
                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, ims[i])
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        ori_ims[i] = draw_bboxes(
                            ori_ims[i], bbox_xyxy, identities, offset=(xmin, ymin)
                        )
                if self.write_video:
                    self.output.write(ori_ims[i])
            self.counter += self.batch_size


if __name__ == "__main__":
    yolo3 = YOLO3("cfg/yolo_v3.cfg", "yolov3.weights", "cfg/coco.names", is_plot=True)
    print("yolo3.size =", yolo3.size)
    import os

    root = "../demo"
    files = [os.path.join(root, file) for file in os.listdir(root)]
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
