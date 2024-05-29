import os
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone

# generate from ChatGPT
coco91_to_coco80 = {
    1: 0,  # 人
    2: 1,  # 自行车
    3: 2,  # 车
    4: 3,  # 摩托车
    5: 4,  # 飞机
    6: 5,  # 巴士
    7: 6,  # 火车
    8: 7,  # 卡车
    9: 8,  # 船
    10: 9,  # 交通灯
    11: 10,  # 消防栓
    13: 11,  # 停止标志
    14: 12,  # 停车米黄线
    15: 13,  # 长凳
    16: 14,  # 鸟
    17: 15,  # 猫
    18: 16,  # 狗
    19: 17,  # 马
    20: 18,  # 羊
    21: 19,  # 牛
    22: 20,  # 大象
    23: 21,  # 熊
    24: 22,  # 斑马
    25: 23,  # 长颈鹿
    27: 24,  # 背包
    28: 25,  # 雨伞
    31: 26,  # 手提包
    32: 27,  # 领带
    33: 28,  # 衬衫
    34: 29,  # 鞋
    35: 30,  # 西装
    36: 31,  # 帽子
    37: 32,  # 袜子
    38: 33,  # 裙子
    39: 34,  # 短裤
    40: 35,  # 手套
    41: 36,  # 围巾
    42: 37,  # 打火机
    43: 38,  # 书
    44: 39,  # 时钟
    46: 40,  # 杯子
    47: 41,  # 刀叉
    48: 42,  # 盘子
    49: 43,  # 水果
    50: 44,  # 螺丝刀
    51: 45,  # 锤子
    52: 46,  # 剪刀
    53: 47,  # 锯
    54: 48,  # 摄像机
    55: 49,  # 电脑
    56: 50,  # 电视
    57: 51,  # 微波炉
    58: 52,  # 烤箱
    59: 53,  # 烤面包机
    60: 54,  # 水槽
    61: 55,  # 冰箱
    62: 56,  # 椅子
    63: 57,  # 镜子
    64: 58,  # 餐桌
    65: 59,  # 网站
    67: 60,  # 面具
    70: 61,  # 枕头
    72: 62,  # 气球
    73: 63,  # 硬盘
    74: 64,  # 风扇
    75: 65,  # 窗户帘
    76: 66,  # 桌子
    77: 67,  # 厕所
    78: 68,  # 锅炉
    79: 69,  # 烟囱
    80: 70,  # 书架
    81: 71,  # 梯子
    82: 72,  # 布艺
    84: 73,  # 画
    85: 74,  # 毯子
    86: 75,  # 沙发
    87: 76,  # 植物
    88: 77,  # 床
    89: 78,  # 镜子
    90: 79,  # 餐车
}


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


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


class Mask_RCNN:
    def __init__(self, segment, num_classes, box_thresh, label_json_path='coco_classes.json', weight_path=None):
        self.segment = segment
        self.num_classes = num_classes  # 不包含背景
        self.box_thresh = box_thresh
        self.weight_path = weight_path
        self.label_json_path = label_json_path
        with open(self.label_json_path, 'r') as f:
            self.category_index = json.load(f)
        self.class_names = [value for value in self.category_index.values()]
        self.category_index = {str(k): v for k, v in enumerate(self.class_names)}
        # get devices
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("using {} device.".format(self.device))

        # create model
        self.model = create_model(num_classes=self.num_classes + 1, box_thresh=self.box_thresh)

        # load train weights
        assert os.path.exists(self.weight_path), "{} file dose not exist.".format(self.weight_path)
        weights_dict = torch.load(self.weight_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        self.model.load_state_dict(weights_dict)
        self.model.to(self.device)

    def __call__(self, img):
        if (img > 1).any():
            img = (img / 255.).astype('float32')
        img = torch.from_numpy(img).permute(2, 0, 1)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img.to(self.device).unsqueeze(0))[0]

        # coco91 to 80
        outputs['labels'] = torch.tensor([coco91_to_coco80[label.item()] for label in outputs['labels']],
                                         device=outputs['boxes'].device)
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
    weight_path = "./save_weights/maskrcnn_resnet50_fpn_coco.pth"

    mask_rcnn = Mask_RCNN(True, num_classes=90, box_thresh=0.5, label_json_path='coco_classes.json', weight_path=weight_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict_boxes, predict_scores, predict_classes, predict_mask = mask_rcnn(img)

    plot_img = draw_objs(Image.fromarray(img),
                         boxes=xywh_to_xyxy(predict_boxes),
                         classes=predict_classes.astype(np.int32),
                         scores=predict_scores,
                         masks=predict_mask,
                         category_index=mask_rcnn.category_index,
                         box_thresh=box_thresh,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
