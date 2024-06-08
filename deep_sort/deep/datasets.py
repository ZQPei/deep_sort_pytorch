import json
import os
import random

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ClsDataset(Dataset):
    def __init__(self, images_path, images_labels, transform=None):
        self.images_path = images_path
        self.images_labels = images_labels
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label = self.images_labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def read_split_data(root, valid_rate=0.2):
    assert os.path.exists(root), 'dataset root: {} does not exist.'.format(root)

    class_names = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    class_names.sort()

    class_indices = {name: i for i, name in enumerate(class_names)}
    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    train_images_path = []
    train_labels = []
    val_images_path = []
    val_labels = []
    per_class_num = []

    supported = ['.jpg', '.JPG', '.png', '.PNG']
    for cls in class_names:
        cls_path = os.path.join(root, cls)
        images_path = [os.path.join(cls_path, i) for i in os.listdir(cls_path)
                       if os.path.splitext(i)[-1] in supported]
        images_label = class_indices[cls]
        per_class_num.append(len(images_path))

        val_path = random.sample(images_path, int(len(images_path) * valid_rate))
        for img_path in images_path:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_labels.append(images_label)
            else:
                train_images_path.append(img_path)
                train_labels.append(images_label)

    print("{} images were found in the dataset.".format(sum(per_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    assert len(train_images_path) > 0, "number of training images must greater than zero"
    assert len(val_images_path) > 0, "number of validation images must greater than zero"

    plot_distribution = False
    if plot_distribution:
        plt.bar(range(len(class_names)), per_class_num, align='center')
        plt.xticks(range(len(class_names)), class_names)

        for i, v in enumerate(per_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('classes')
        plt.ylabel('numbers')
        plt.title('the distribution of dataset')
        plt.show()
    return [train_images_path, train_labels], [val_images_path, val_labels], len(class_names)
