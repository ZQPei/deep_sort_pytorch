from .YOLOv3 import YOLOv3


__all__ = ['build_detector']

def build_detector(args, use_cuda):
    return YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
