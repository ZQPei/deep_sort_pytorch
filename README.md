# Deep Sort with PyTorch

## Latest Update(07-22)
Changes
- bug fix (Thanks @JieChen91 and @yingsen1 for bug reporting).  
- using batch for feature extracting for each frame, which lead to a small speed up.  
- code improvement.

Futher improvement direction  
- Train detector on specific dataset rather than the official one.
- Retrain REID model on pedestrain dataset for better performance.
- Replace YOLOv3 detector with advanced ones.

Any contributions to this repository is welcome!

![](images/demo.gif)


## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

## Dependencies
- python 3 (python2 not sure)
- numpy
- scipy
- opencv-python
- sklearn
- pytorch 0.4 or 1.x

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
```
git clone git@github.com:ZQPei/deep_sort_pytorch.git
```

2. Download YOLOv3 parameters
```
cd YOLOv3/
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```

3. Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from 
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

4. Run demo
```
usage: demo_yolo3_deepsort.py VIDEO_PATH
                              [--help] 
                              [--yolo_cfg YOLO_CFG]
                              [--yolo_weights YOLO_WEIGHTS]
                              [--yolo_names YOLO_NAMES]
                              [--conf_thresh CONF_THRESH]
                              [--nms_thresh NMS_THRESH]
                              [--deepsort_checkpoint DEEPSORT_CHECKPOINT]
                              [--max_dist MAX_DIST] [--ignore_display]
                              [--display_width DISPLAY_WIDTH]
                              [--display_height DISPLAY_HEIGHT]
                              [--save_path SAVE_PATH]          
```

All files above can also be accessed from BaiduDisk!  
linker：https://pan.baidu.com/s/1TEFdef9tkJVT0Vf0DUZvrg  
passwd：1eqo  

## Training the RE-ID model
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

To train the model, first you need download [Market1501](http://www.liangzheng.org/Project/project_reid.html) dataset or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.  

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).
![train.jpg](deep_sort/deep/train.jpg)

## Demo videos and images
[demo.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
[demo2.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)

![1.jpg](images/1.jpg)
![2.jpg](images/2.jpg)


## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)



