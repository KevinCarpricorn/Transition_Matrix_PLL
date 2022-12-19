import numpy as np
import torch
import torch.nn.functional as F


def partial_loss(output1, target):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY

    return loss, new_target


def evaluate(args, loader, model):
    model.eval()
    correct = 0
    total = 0
    for images, _, labels, _ in loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        output1 = model(images)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


def norm(T):
    min_ = torch.min(T, dim=2, keepdim=True)[0]
    T = T - min_
    T = T / (torch.max(T, dim=1, keepdim=True)[0] - min_ + 1e-8)
    for i in range(T.size(0)):
        for j in range(T.size(1)):
            T[i, j, j] = 1
    return T


def accuracy(output, target, topk=(1,)):
    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
