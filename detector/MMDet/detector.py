import logging
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector
from .mmdet_utils import xyxy_to_xywh

class MMDet(object):
    def __init__(self, cfg_file, checkpoint_file, score_thresh=0.7,
                is_xywh=False, use_cuda=True):
        # net definition
        self.device = "cuda" if use_cuda else "cpu"
        self.net = init_detector(cfg_file, checkpoint_file, device=self.device)
        logger = logging.getLogger("root.detector")
        logger.info('Loading weights from %s... Done!' % (checkpoint_file))

        #constants
        self.score_thresh = score_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.class_names = self.net.CLASSES
        self.num_classes = len(self.class_names)

    def __call__(self, ori_img):
        # forward
        bbox_result = inference_detector(self.net, ori_img)
        bboxes = np.vstack(bbox_result)

        if len(bboxes) == 0:
            bbox = np.array([]).reshape([0, 4])
            cls_conf = np.array([])
            cls_ids = np.array([])
            return bbox, cls_conf, cls_ids

        bbox = bboxes[:, :4]
        cls_conf = bboxes[:, 4]
        cls_ids = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        cls_ids = np.concatenate(cls_ids)

        selected_idx = cls_conf > self.score_thresh
        bbox = bbox[selected_idx, :]
        cls_conf = cls_conf[selected_idx]
        cls_ids = cls_ids[selected_idx]

        if self.is_xywh:
            bbox = xyxy_to_xywh(bbox)

        return bbox, cls_conf, cls_ids


            

