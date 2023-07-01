import torch,torchvision
import cv2
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
device = "cuda" if torch.cuda.is_available() else "cpu"
maskrcnn = maskrcnn.to(device)
maskrcnn.eval()

def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh

class mask_rcnn():
    def __init__(self,segment):
        self.segment = segment
    def __call__(self,img):

        if (img>1).any():
            img=(img/255).astype('float32')
        print(f"{img.shape=}",type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        o = maskrcnn([img.to(device)])[0]
        if self.segment:
            return (xyxy_to_xywh(o['boxes']).cpu().detach().numpy(),
        o['scores'].cpu().detach().numpy(),o['labels'].cpu().detach().numpy(),o['masks'].cpu().detach().numpy())
        else:
            return (xyxy_to_xywh(o['boxes']).cpu().detach().numpy(),
        o['scores'].cpu().detach().numpy(),o['labels'].cpu().detach().numpy())


