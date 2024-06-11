In deepsort algorithm, appearance feature extraction network used to extract features from **image_crops** for matching purpose.The original model used in paper is in `model.py`, and its parameter here [ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6). This repository also provides a `resnet.py` script and its pre-training weights on Imagenet here.

```
# resnet18
https://download.pytorch.org/models/resnet18-5c106cde.pth
# resnet34    
https://download.pytorch.org/models/resnet34-333f7ec4.pth
# resnet50
https://download.pytorch.org/models/resnet50-19c8e357.pth
# resnext50_32x4d
https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
```

## Dataset PrePare

To train the model, first you need download [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.  

If you want to train on your **own dataset**, assuming you have already downloaded the dataset.The dataset should be arranged in the following way.

```
 ├── dataset_root: The root dir of the dataset.
  ├── class1: Category 1 is located in the folder dir.
   ├── xxx1.jpg: Image belonging to category 1.
   ├── xxx2.jpg: Image belonging to category 1.
  ├── class2: Category 2 is located in the folder dir.
   ├── xxx3.jpg: Image belonging to category 2.
   ├── xxx4.jpg: Image belonging to category 2.
  ├── class3: Category 3 is located in the folder dir.
  ...
  ...
```

## Training the RE-ID model

Assuming you have already prepare the dataset. Then you can use the following command to start your training progress.

#### training on a single GPU

```python
usage: train.py [--data-dir]
                [--epochs]
                [--batch_size]
                [--lr]
                [--lrf]
                [--weights]
                [--freeze-layers]
                [--gpu_id]

# default use cuda:0, use Net in `model.py`
python train.py --data-dir [dataset/root/path] --weights [(optional)pre-train/weight/path]
# you can use `--freeze-layers` option to freeze full convolutional layer parameters except fc layers parameters
python train.py --data-dir [dataset/root/path] --weights [(optional)pre-train/weight/path] --freeze-layers
```

#### training on multiple GPU

```python
usage: train_multiGPU.py [--data-dir]
                         [--epochs]
                         [--batch_size]
                         [--lr]
                         [--lrf]
                         [--syncBN]
                         [--weights]
                         [--freeze-layers]
                         # not change the following parameters, the system will automatically assignment
                         [--device]
                         [--world_size]
                         [--dist_url]
                         
# default use cuda:0, cuda:1, cuda:2, cuda:3, use resnet18 in `resnet.py`
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_multiGPU.py --data-dir [dataset/root/path] --weights [(optional)pre-train/weight/path] 
# you can use `--freeze-layers` option to freeze full convolutional layer parameters except fc layers parameters
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_multiGPU.py --data-dir [dataset/root/path] --weights [(optional)pre-train/weight/path] --freeze-layers
```

An example of training progress is as follows:

![train.jpg](./train.jpg)

The last, you can evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).

