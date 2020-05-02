import os
from os.path import splitext, basename, join, dirname, isdir, isfile

import cv2
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from warnings import warn

from detector import build_detector
from deep_sort import build_tracker
from utils.tools import tik_tok, is_video
from utils.tools import compute_color_for_labels
from utils.parser import get_config
from utils.bbox_to_json import JsonBboxLogger


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.json_logger = JsonBboxLogger()
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warn(UserWarning("Running in cpu mode!"))

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cv2.VideoCapture.get(self.vdo, cv2.CAP_PROP_FRAME_COUNT))
        video_details = {'frame_width': self.im_width,
                         'frame_height': self.im_height,
                         'frame_rate': self.args.write_fps,
                         'video_name': self.args.input}
        self.json_logger.add_video_details(**video_details)

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            filename, extension = splitext(basename(self.args.VIDEO_PATH))
            self.output_file = join(self.args.save_path, f'{filename}.avi')
            self.json_output = join(self.args.save_path, f'{filename}.json')
            if not isdir(dirname(self.json_output)):
                os.makedirs(dirname(self.json_output))
            self.writer = cv2.VideoWriter(self.args.output_file, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        pbar = tqdm(total=self.total_frames + 1)
        while self.vdo.grab():
            if idx_frame % args.frame_interval == 0:
                _, ori_im = self.vdo.retrieve()
                cv2_time = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
                cv2_frame_idx = int(self.vdo.get(cv2.CAP_PROP_POS_FRAMES))
                self.json_logger.add_frame(frame_id=cv2_frame_idx, timestamp=cv2_time)
                self.detection(frame=ori_im, frame_id=cv2_frame_idx)

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                idx_frame += 1
            pbar.update()
        self.json_logger.json_output(self.json_output)

    @tik_tok
    def detection(self, frame, frame_id, inplace=True):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        if bbox_xywh is not None:
            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                frame = self.draw_boxes(img=frame, frame_id=frame_id, output=outputs)

            if not inplace:
                return frame

    def draw_boxes(self, img, frame_id, output, offset=(0, 0)):
        for i, box in enumerate(output):
            x1, y1, x2, y2, identity = [int(ii) for ii in box]
            self.json_logger.add_bbox_to_frame(frame_id=frame_id,
                                          bbox_id=identity,
                                          top=y1,
                                          left=x1,
                                          width=x2 - x1,
                                          height=y2 - y1)
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # box text and bar
            self.json_logger.add_label_to_bbox(frame_id=frame_id, bbox_id=identity, category='pedestrian', confidence=0.9)
            color = compute_color_for_labels(identity)
            label = '{}{:d}'.format("", identity)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    args = parser.parse_args()

    assert isfile(args.VIDEO_PATH), "Error: Unable to find the video file"
    assert is_video(args.VIDEO_PATH), "Error: Video format is not supported"
    if args.frame_interval < 1: args.frame_interval = 1
    # Detection system gets confused if frame_interval get more than 7
    elif args.frame_interval > 10: args.frame_interval = 7
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
