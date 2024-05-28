import json
import os

import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from matplotlib import pyplot as plt


mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mask_rcnn.to(device)
mask_rcnn.eval()


def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh

def xywh_to_xyxy(boxes_xywh):
    if isinstance(boxes_xywh, torch.Tensor):
        boxes_xyxy = boxes_xywh.clone()
    elif isinstance(boxes_xywh, np.ndarray):
        boxes_xyxy = boxes_xywh.copy()

    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    return boxes_xyxy


class MaskRCNN:
    def __init__(self, segment):
        self.segment = segment
        self.class_names = ['person', 'bicycle']

    def __call__(self, img):
        if (img > 1).any():
            img = (img / 255.).astype('float32')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        outputs = mask_rcnn([img.to(device)])[0]

        if self.segment:
            return (xyxy_to_xywh(outputs['boxes']).detach().cpu().numpy(),
                    outputs['scores'].detach().cpu().numpy(),
                    outputs['labels'].detach().cpu().numpy(),
                    outputs['masks'].squeeze().detach().cpu().numpy())
        else:
            return (xyxy_to_xywh(outputs['boxes']).detach().cpu().numpy(),
                    outputs['scores'].detach().cpu().numpy(),
                    outputs['labels'].detach().cpu().numpy())


if __name__ == '__main__':
    from draw_box_utils import draw_objs

    num_classes = 90  # 不包含背景
    box_thresh = 0.5
    img_path = "./test.jpg"
    label_json_path = './coco91_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    Mask_Rcnn = MaskRCNN(True)
    img = cv2.imread(img_path)
    predict_boxes, predict_scores, predict_classes, predict_mask = Mask_Rcnn(img)

    plot_img = draw_objs(Image.fromarray(img[:, :, ::-1]),
                         boxes=xywh_to_xyxy(predict_boxes),
                         classes=predict_classes.astype(np.int32),
                         scores=predict_scores,
                         masks=predict_mask,
                         category_index=category_index,
                         box_thresh=box_thresh,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
