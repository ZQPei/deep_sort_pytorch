import argparse
import os
import tempfile

import math
import warnings
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.optim import lr_scheduler

from multi_train_utils.distributed_utils import init_distributed_mode, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate, load_model
import torch.distributed as dist
from datasets import ClsDataset, read_split_data

from model import Net
from resnet import resnet18

# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1_err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


def main(args):
    batch_size = args.batch_size
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'

    train_info, val_info, num_classes = read_split_data(args.data_dir, valid_rate=0.2)
    train_images_path, train_labels = train_info
    val_images_path, val_labels = val_info

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((128, 64), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ClsDataset(
        images_path=train_images_path,
        images_labels=train_labels,
        transform=transform_train
    )
    val_dataset = ClsDataset(
        images_path=val_images_path,
        images_labels=val_labels,
        transform=transform_val
    )

    number_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(number_workers))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=number_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=number_workers,
    )

    # net definition
    start_epoch = 0
    net = Net(num_classes=num_classes)
    if args.weights:
        print('Loading from ', args.weights)
        checkpoint = torch.load(args.weights, map_location='cpu')
        net_dict = checkpoint if 'net_dict' not in checkpoint else checkpoint['net_dict']
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else start_epoch
        net = load_model(net_dict, net.state_dict(), net)

    if args.freeze_layers:
        for name, param in net.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    net.to(device)

    # loss and optimizer
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, args.lr, momentum=0.9, weight_decay=5e-4)

    lr = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_positive, train_loss = train_one_epoch(net, optimizer, train_loader, device, epoch)
        train_acc = train_positive / len(train_dataset)
        scheduler.step()

        test_positive, test_loss = evaluate(net, val_loader, device)
        test_acc = test_positive / len(val_dataset)

        print('[epoch {}] accuracy: {}'.format(epoch, test_acc))

        state_dict = {
            'net_dict': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch
        }
        torch.save(state_dict, './checkpoint/model_{}.pth'.format(epoch))
        draw_curve(epoch, train_loss, 1 - train_acc, test_loss, 1 - test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='data', type=str)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--lrf', default=0.1, type=float)

    parser.add_argument('--weights', type=str, default='./checkpoint/resnet18.pth')
    parser.add_argument('--freeze-layers', action='store_true')

    parser.add_argument('--gpu_id', default='0', help='gpu id')
    args = parser.parse_args()

    main(args)
