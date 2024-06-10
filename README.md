# Deep Sort with PyTorch

![](demo/demo.gif)

## Update(1-1-2020)
Changes
- fix bugs
- refactor code
- accerate detection by adding nms on gpu

## Update(07-22)
Changes
- bug fix (Thanks @JieChen91 and @yingsen1 for bug reporting).  
- using batch for feature extracting for each frame, which lead to a small speed up.  
- code improvement.

Futher improvement direction  
- Train detector on specific dataset rather than the official one.
- Retrain REID model on pedestrain dataset for better performance.
- Replace YOLOv3 detector with advanced ones.

## Update(23-05-2024)

### tracking 

- Added resnet network to the appearance feature extraction network in the deep folder

- Fixed the NMS bug in the `preprocessing.py` and also fixed covariance calculation bug in the `kalmen_filter.py` in the sort folder

### detecting

- Added YOLOv5 detector, aligned interface, and added YOLOv5 related yaml configuration files. Codes references this repo: [YOLOv5-v6.1](https://github.com/ultralytics/yolov5/tree/v6.1).

- The `train.py`, `val.py` and `detect.py` in the original YOLOv5 were deleted. This repo only need **yolov5x.pt**.

### deepsort

- Added tracking target category, which can display both category and tracking ID simultaneously.

## Update(28-05-2024)

### segmentation

* Added Mask RCNN instance segmentation model. Codes references this repo: [mask_rcnn](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/mask_rcnn). Visual result saved in `demo/demo2.gif`.
* Similar to YOLOv5, `train.py`, `validation.py` and `predict.py` were deleted. This repo only need **maskrcnn_resnet50_fpn_coco.pth**.

### deepsort

- Added tracking target mask, which can display both category, tracking ID and target mask simultaneously.

## latest Update(09-06-2024)

### feature extraction network

* Using `nn.parallel.DistributedDataParallel` in PyTorch to support multiple GPUs training.
* Added [GETTING_STARTED.md](deep_sort/deep/GETTING_STARTED.md) for better using `train.py` and `train_multiGPU.py`.

Updated `README.md` for previously updated content(#Update(23-05-2024) and #Update(28-05-2024)).

**Any contributions to this repository is welcome!**


## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

## Dependencies
- python 3 **(python2 not sure)**
- numpy
- scipy
- opencv-python
- sklearn
- torch >= 1.9
- torchvision >= 0.13
- pillow
- vizer
- edict
- matplotlib
- pycocotools
- tqdm

## Quick Start
0. Check all dependencies installed
```bash
pip install -r requirements.txt
```
for user in china, you can specify pypi source to accelerate install like:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

1. Clone this repository
```bash
git clone git@github.com:ZQPei/deep_sort_pytorch.git
```

2. Download detector parameters
```bash
# if you use YOLOv3 as detector in this repo
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../

# if you use YOLOv5 as detector in this repo
cd detector/YOLOv5
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
or 
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
cd ../../

# if you use Mask RCNN as detector in this repo
cd detector/Mask_RCNN/save_weights
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
cd ../../../
```

3. Download deepsort feature extraction networks weight
```bash
# if you use original model in PAPER
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../

# if you use resnet18 in this repo
cd deep_sort/deep/checkpoint
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
cd ../../../
```

4. **(Optional)** Compile nms module if you use YOLOv3 as detetor in this repo
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

Notice:
If compiling failed, the simplist way is to **Upgrade your pytorch >= 1.1 and torchvision >= 0.3" and you can avoid the troublesome compiling problems which are most likely caused by either `gcc version too low` or `libraries missing`.

5. **(Optional)** Prepare third party submodules

[fast-reid](https://github.com/JDAI-CV/fast-reid)

This library supports bagtricks, AGW and other mainstream ReID methods through providing an fast-reid adapter.

to prepare our bundled fast-reid, then follow instructions in its README to install it.

Please refer to `configs/fastreid.yaml` for a sample of using fast-reid. See [Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/docs/MODEL_ZOO.md) for available methods and trained models.

[MMDetection](https://github.com/open-mmlab/mmdetection)

This library supports Faster R-CNN and other mainstream detection methods through providing an MMDetection adapter.

to prepare our bundled MMDetection, then follow instructions in its README to install it.

Please refer to `configs/mmdet.yaml` for a sample of using MMDetection. See [Model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) for available methods and trained models.

Run

```
git submodule update --init --recursive
```


6. Run demo
```bash
usage: deepsort.py [-h]
                   [--fastreid]
                   [--config_fastreid CONFIG_FASTREID]
                   [--mmdet]
                   [--config_mmdetection CONFIG_MMDETECTION]
                   [--config_detection CONFIG_DETECTION]
                   [--config_deepsort CONFIG_DEEPSORT] [--display]
                   [--frame_interval FRAME_INTERVAL]
                   [--display_width DISPLAY_WIDTH]
                   [--display_height DISPLAY_HEIGHT] [--save_path SAVE_PATH]
                   [--cpu] [--camera CAM]
                   VIDEO_PATH         

# yolov3 + deepsort
python deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3.yaml

# yolov3_tiny + deepsort
python deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml

# yolov3 + deepsort on webcam
python3 deepsort.py /dev/video0 --camera 0

# yolov3_tiny + deepsort on webcam
python3 deepsort.py /dev/video0 --config_detection ./configs/yolov3_tiny.yaml --camera 0

# yolov5s + deepsort
python deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov5s.yaml

# yolov5m + deepsort
python deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov5m.yaml

# mask_rcnn + deepsort
python deepsort.py [VIDEO_PATH] --config_detection ./configs/mask_rcnn.yaml --segment

# fast-reid + deepsort
python deepsort.py [VIDEO_PATH] --fastreid [--config_fastreid ./configs/fastreid.yaml]

# MMDetection + deepsort
python deepsort.py [VIDEO_PATH] --mmdet [--config_mmdetection ./configs/mmdet.yaml]
```
Use `--display` to enable display image per frame.  
Results will be saved to `./output/results.avi` and `./output/results.txt`.

All files above can also be accessed from BaiduDisk!  
linker：[BaiduDisk](https://pan.baidu.com/s/1YJ1iPpdFTlUyLFoonYvozg)
passwd：fbuw

## Training the RE-ID model
Check [GETTING_STARTED.md](deep_sort/deep/GETTING_STARTED.md) to start training progress using standard benchmark or **customized dataset**.

## Demo videos and images
[demo.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
[demo2.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)

![1.jpg](demo/1.jpg)
![2.jpg](demo/2.jpg)


## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- paper: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
- paper: [Mask R-CNN](https://arxiv.org/pdf/1703.06870)
- code: [WZMIAOMIAO/Mask R-CNN](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/mask_rcnn)
- paper: [YOLOv5](https://github.com/ultralytics/yolov5)
- code: [ultralytics/yolov5](https://github.com/ultralytics/yolov5/tree/v6.1)
