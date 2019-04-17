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
    parser.add_argument("-path", "--videos_path", type=str)
    parser.add_argument("-opath", "--output_path", type=str)
    return parser.parse_args(args)


class loading(StoppableThread):
    def __init__(self, **kwargs):
        StoppableThread.__init__(self)
        self.__dict__.update(**kwargs)
        self.is_opened = False
        self.vdo = cv2.VideoCapture()

    def run(self):
        while len(self.videos) != 0:
            self.open()
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
                    imgs = np.array(
                        list(map(lambda img: cv2.resize(img, self.size), imgs))
                    )
                    imgs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2)
                    if self.half:
                        imgs = imgs.half()

                    model_queue.append(
                        [
                            ori_ims,
                            ims,
                            imgs,
                            self.area,
                            self.im_width,
                            self.im_height,
                            self.video_read,
                        ]
                    )
                    ims, ori_ims = [], []

    def open(self):
        assert os.path.isfile(video_path), "Error: path error"
        self.video_read = self.videos.pop()
        self.vdo.open(self.videos_path + self.video_read)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height


class Detector(StoppableThread):
    def __init__(self, **kwargs):
        StoppableThread.__init__(self)
        self.__dict__.update(**kwargs)
        self.yolo3 = YOLO3(
            "YOLO3/cfg/yolo_v3.cfg",
            "YOLO3/yolov3.weights",
            "YOLO3/cfg/coco.names",
            is_xywh=True,
        )
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True
        self.output = None
        self.counter = 0

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
                video_read,
            ) = self.model_queue.pop(0)
            if self.write_video and self.video_read != video_read:
                self.output = None

            xmin, ymin, xmax, ymax = area
            if self.write_video and self.output is None:
                self.video_read = videao_read
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(
                    self.output_path + video_read,
                    fourcc,
                    20,
                    (self.im_width, self.im_height),
                )
            batch_bbox_xywh, batch_cls_conf, batch_cls_ids = self.yolo3(imgs, ori_ims)

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
    import sys

    config = parse_args(sys.argv[1:])
    batch_size = config.batch_size
    print(config)
    videos = os.listdir(config.videos_path)
    model_queue = []
    det = threadTask(
        Detector,
        model_queue=model_queue,
        half=config.half,
        batch_size=batch_size,
        output_path=config.output_path,
    )
    loading_thread = threadTask(
        loading,
        model_queue=model_queue,
        batch_size=batch_size,
        size=det.task.yolo3.size,
        half=config.half,
        videos=videos,
        videos_path=config.videos_path,
    )

    start = time.time()
    old_value = 0
    while 1:
        time.sleep(10)

        end = time.time()
        new_value = det.task.counter
        print(
            "time: {}s, fps: {}".format(
                end - start, (new_value - old_value) / (end - start)
            ),
            len(model_queue),
        )
        old_value = new_value
        start = time.time()
