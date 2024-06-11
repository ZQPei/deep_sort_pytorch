from .YOLOv3 import YOLOv3
from .MMDet import MMDet
from .YOLOv5 import YOLOv5
from .Mask_RCNN import Mask_RCNN

__all__ = ['build_detector']


def build_detector(cfg, use_cuda, segment=False):
    if cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                     score_thresh=cfg.MMDET.SCORE_THRESH,
                     is_xywh=True, use_cuda=use_cuda)
    elif cfg.USE_SEGMENT:
        return Mask_RCNN(segment, num_classes=cfg.MASKRCNN.NUM_CLASSES, box_thresh=cfg.MASKRCNN.BOX_THRESH,
                         label_json_path=cfg.MASKRCNN.LABEL, weight_path=cfg.MASKRCNN.WEIGHT)
    else:
        if 'YOLOV5' in cfg:
            return YOLOv5(cfg.YOLOV5.WEIGHT, cfg.YOLOV5.DATA, cfg.YOLOV5.IMGSZ,
                          cfg.YOLOV5.SCORE_THRESH, cfg.YOLOV5.NMS_THRESH, cfg.YOLOV5.MAX_DET)
        elif 'YOLOV3' in cfg:
            return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                          score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                          is_xywh=True, use_cuda=use_cuda)
