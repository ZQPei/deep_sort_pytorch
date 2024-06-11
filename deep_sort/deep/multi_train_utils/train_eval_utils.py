import sys

from tqdm import tqdm
import torch

from .distributed_utils import reduce_value, is_main_process


def load_model(state_dict, model_state_dict, model):
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape {}, ' \
                      'loaded shape {}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for idx, (images, labels) in enumerate(data_loader):
        # forward
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * idx + loss.detach()) / (idx + 1)
        pred = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()

        if is_main_process():
            data_loader.desc = '[epoch {}] mean loss {}'.format(epoch, mean_loss.item())

        if not torch.isfinite(loss):
            print('loss is infinite, ending training')
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(), mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = reduce_value(loss, average=True)

        test_loss = (test_loss * idx + loss.detach()) / (idx + 1)
        pred = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()

    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(), test_loss.item()
