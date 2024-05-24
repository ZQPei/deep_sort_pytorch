from .YOLOv3 import YOLOv3
from .MMDet import MMDet
from .YOLOv5 import YOLOv5

__all__ = ['build_detector']

def build_detector(cfg, use_cuda):
    if cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                    score_thresh=cfg.MMDET.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    else:
        if 'YOLOV5' in cfg:
            return YOLOv5(cfg.YOLOV5.WEIGHT, cfg.YOLOV5.DATA, cfg.YOLOV5.IMGSZ,
                          cfg.YOLOV5.SCORE_THRESH, cfg.YOLOV5.NMS_THRESH, cfg.YOLOV5.MAX_DET)
        elif 'YOLOv3' in cfg:
            return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                        score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                        is_xywh=True, use_cuda=use_cuda)
