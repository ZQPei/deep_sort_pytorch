import os
import cv2
import numpy as np
import torch
from YOLO3 import YOLO3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time
from algorithmes.common.pipeline import threadTask, StoppableThread
import configargparse
from yolo_utils import get_all_boxes, nms, plot_boxes_cv2


def parse_args(args):
    parser = configargparse.ArgumentParser(
        default_config_files=[""], auto_env_var_prefix="veesion_"
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=32)

    parser.add_argument("-half", action="store_true", help="FP16 inference")
    parser.add_argument("-path", "--videos_path", type=str, default=60)
    return parser.parse_args(args)


class loading(StoppableThread):
    def __init__(self, **kwargs):
        StoppableThread.__init__(self)
        self.__dict__.update(**kwargs)
        self.is_opened = False
        self.vdo = cv2.VideoCapture()

    def run(self):
        while not self.is_opened:
            time.sleep(1.0)
        xmin, ymin, xmax, ymax = self.area
        ims, ori_ims = [], []
        while self.vdo.grab():
            _, ori_im = self.vdo.retrieve()
            ori_ims.append(ori_im)
            ims.append(ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)])
            if len(ims) == self.batch_size:
                while len(model_queue) > 5:
                    time.sleep(0.2)
                assert isinstance(
                    ori_ims[0], np.ndarray
                ), "input must be a numpy array!"
                imgs = list(
                    map(lambda ori_img: ori_img.astype(np.float) / 255.0, ori_ims)
                )
                imgs = np.array(list(map(lambda img: cv2.resize(img, self.size), imgs)))
                imgs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2)
                if self.half:
                    imgs = imgs.half()

                model_queue.append(
                    [ori_ims, ims, imgs, self.area, self.im_width, self.im_height]
                )
                ims, ori_ims = [], []

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        self.is_opened = self.vdo.isOpened()


class Detector(StoppableThread):
    def __init__(self, **kwargs):
        StoppableThread.__init__(self)
        self.__dict__.update(**kwargs)
        self.yolo3 = YOLO3(
            "YOLO3/cfg/yolo_v3.cfg",
            "YOLO3/yolov3.weights",
            "YOLO3/cfg/coco.names",
            is_xywh=True,
            half=self.half,
        )
        self.write_video = True

    def run(self):
        while 1:
            while len(self.model_queue) == 0:
                time.sleep(0.2)
            (
                ori_ims,
                ims,
                imgs,
                area,
                self.im_width,
                self.im_height,
            ) = self.model_queue.pop(0)

            xmin, ymin, xmax, ymax = area
            batch_boxes = self.yolo3(imgs)
            self.postprocessing_queue.append(
                [batch_boxes, ims, ori_ims, area, self.im_width, self.im_height]
            )


class postProcessing(StoppableThread):
    def __init__(self, **kwargs):
        StoppableThread.__init__(self)
        self.__dict__.update(**kwargs)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.write_video = True
        self.output = None
        self.counter = 0
        # net definition
        # constants
        self.use_cuda = True
        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        self.is_plot = False
        self.is_xywh = False
        self.class_names = self.load_class_names("YOLO3/cfg/coco.names")

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

    def postprocess(self, batch_boxes, ori_ims):

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


if __name__ == "__main__":
    import sys

    config = parse_args(sys.argv[1:])
    batch_size = config.batch_size
    print(config)
    model_queue, postprocessing_queue = [], []
    det = threadTask(
        Detector,
        model_queue=model_queue,
        half=config.half,
        postprocessing_queue=postprocessing_queue,
        batch_size=batch_size,
    )
    loading_thread = threadTask(
        loading,
        model_queue=model_queue,
        batch_size=batch_size,
        size=det.task.yolo3.size,
        half=config.half,
    )
    loading_thread.task.open(config.videos_path)

    postprocessing_thread = threadTask(
        postProcessing,
        model_queue=model_queue,
        postprocessing_queue=postprocessing_queue,
        batch_size=batch_size,
        num_classes=det.task.yolo3.net.num_classes,
    )

    start = time.time()
    old_value = 0
    while 1:
        time.sleep(10)

        end = time.time()
        new_value = postprocessing_thread.task.counter
        print(
            "time: {}s, fps: {}".format(
                end - start, (new_value - old_value) / (end - start)
            ),
            len(model_queue),
            len(postprocessing_queue),
        )
        old_value = new_value
        start = time.time()
