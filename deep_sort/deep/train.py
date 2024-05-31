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
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
import torch.distributed as dist

from model import Net
from resnet import resnet18

# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


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
    init_distributed_mode(args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size
    checkpoint_path = ''

    if rank == 0:
        print(args)
        if os.path.exists('./checkpoint') is False:
            os.mkdir('./checkpoint')

    # data loading
    root = args.data_dir
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((128, 64), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    number_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    if rank == 0:
        print('Using {} dataloader workers every process'.format(number_workers))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        pin_memory=True,
        shuffle=True,
        num_workers=number_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=number_workers,
        shuffle=False
    )

    num_classes = max(len(train_loader.dataset.classes), len(test_loader.dataset.classes))

    # net definition
    start_epoch = 0
    net = Net(num_classes=num_classes)
    if args.weights:
        warnings.warn("better providing pretraining weights")
        print('Loading from ', args.weights)
        checkpoint = torch.load(args.weights)
        net_dict = checkpoint if 'net_dict' not in checkpoint else checkpoint['net_dict']
        net.load_state_dict(net_dict)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weights.pth')
        if rank == 0:
            torch.save(net.state_dict(), checkpoint_path)

        dist.barrier()
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    net.to(device)

    if args.freeze_layers:
        for name, param in net.named_parameters():
            if name not in ['fc', 'classifier']:
                param.requires_grad = False
    else:
        if args.syncBN:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    # loss and optimizer
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, args.lr, momentum=0.9, weight_decay=5e-4)

    lr = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_pos, train_loss = train_one_epoch(net, optimizer, train_loader, device, epoch)
        train_acc = train_pos / len(train_dataset)
        scheduler.step()

        test_pos, test_loss = evaluate(net, test_loader, device)
        test_acc = test_pos / len(test_dataset)

        if rank == 0:
            print('[epoch {}] accuracy: {}'.format(epoch, test_acc.item()))

            state_dict = {
                'net_dict': net.module.state_dict(),
                'acc': test_acc,
                'epoch': epoch
            }
            torch.save(state_dict, './checkpoint/model_{}.pth'.format(epoch))
        draw_curve(epoch, train_loss, 1 - train_acc, test_loss, 1 - test_acc)

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='data', type=str)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument('--lrf', default=0.001, type=float)
    parser.add_argument('--syncBN', type=bool, default=True)

    parser.add_argument('--weights', type=str, default='./checkpoint/ckpt.t7')
    parser.add_argument('--freeze_layers', type=bool, default=False)

    # not change the following parameters, the system will automatically assignment
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0, 1 or cpu)')
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    main(args)
