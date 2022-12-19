import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import argparse, sys
import numpy as np
import datasets.distilled_datasets
from collections import OrderedDict
from datasets.MNIST import mnist
from datasets.Fashion_MNIST import fashion
from datasets.CIFAR10 import cifar10
from torch.utils.data import DataLoader
from resnet import ResNet18_M, ResNet34
from resnet_bayes import Bayes_ResNet18_M, Bayes_ResNet34
from utils.utils_algo import *
from loss import LSEP

parser = argparse.ArgumentParser(description='Bayes Transition matrix partial-label learning')
parser.add_argument('--dataset', default='fashion_mnist', type=str, help='dataset', choices=['mnist', 'fashion_mnist', 'cifar10'])
parser.add_argument('--print', default=False, type=bool, help='print')
parser.add_argument('--print_freq', default=100, type=int, help='print_freq')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--classes', default=10, type=int, help='class')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, help='epochs')
parser.add_argument('--bayes_epochs', default=50, type=int, help='warmup epochs')
parser.add_argument('--warmup_epochs', default=20, type=int, help='warmup epochs')
parser.add_argument('--partial_type', help='flipping strategy', type=str, default='binomial', choices=['binomial', 'pair'])
parser.add_argument('--flipping_rate', default=0.1, type=float, help='flipping rate', choices=[0.1, 0.3, 0.5, 0.7])
parser.add_argument('--result_dir', type=str, default='./results', help='result saving dir')
parser.add_argument('--revision', type=bool, default=True, help='revision')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--rho', type=float, default=0.3, help='upper bound of flipping_rate')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--validation_rate', type=float, default=0.1, help='validation_rate')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def main(args):
    print(args)
    model_dir = os.path.join(args.result_dir, f'type:{args.partial_type}_rate:{args.flipping_rate}_rho:{args.rho}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load dataset
    print('==> Preparing data..')
    if args.dataset == 'mnist':
        train_dataset = mnist(root='./datasets/mnist/',
                              download=True,
                              train=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                              partial_type=args.partial_type,
                              partial_rate=args.flipping_rate
                              )
        train_dataset_temp = mnist(root='./datasets/mnist/',
                              download=True,
                              train=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                              partial_type=args.partial_type,
                              partial_rate=args.flipping_rate
                              )
        test_dataset = mnist(root='./datasets/mnist/',
                             download=True,
                             train=False,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                             partial_type=args.partial_type,
                             partial_rate=args.flipping_rate
                             )
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, drop_last=False, shuffle=True)
        train_loader_temp = DataLoader(dataset=train_dataset_temp, batch_size=args.batch_size,
                                            num_workers=args.num_workers, drop_last=False, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False)
        for i in range(len(train_dataset)):
            for j in range(args.classes):
                if train_loader_temp.dataset.train_final_labels[i, j] > 0:
                    train_loader_temp.dataset.train_final_labels[i, j] = 1
        model = ResNet18_M(num_classes=args.classes).to(args.device)
    elif args.dataset == 'fashion_mnist':
        train_dataset = fashion(root='./datasets/fashion_mnist/',
                                download=True,
                                train=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                partial_type=args.partial_type,
                                partial_rate=args.flipping_rate
                                )
        train_dataset_temp = fashion(root='./datasets/fashion_mnist/',
                                download=True,
                                train=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                partial_type=args.partial_type,
                                partial_rate=args.flipping_rate
                                )
        test_dataset = fashion(root='./datasets/fashion_mnist/',
                               download=True,
                               train=False,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.flipping_rate
                               )
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, drop_last=False, shuffle=True)
        train_loader_temp = DataLoader(dataset=train_dataset_temp, batch_size=args.batch_size,
                                            num_workers=args.num_workers, drop_last=False, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False)
        for i in range(len(train_dataset)):
            for j in range(args.classes):
                if train_loader_temp.dataset.train_final_labels[i, j] > 0:
                    train_loader_temp.dataset.train_final_labels[i, j] = 1
        model = ResNet18_M(num_classes=args.classes).to(args.device)
    elif args.dataset == 'cifar10':
        train_dataset = cifar10(root='./datasets/cifar10/',
                                download=True,
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                   (0.2023, 0.1994, 0.2010)), ]),
                                partial_type=args.partial_type,
                                partial_rate=args.flipping_rate
                                )
        train_dataset_temp = cifar10(root='./datasets/cifar10/',
                                download=True,
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                   (0.2023, 0.1994, 0.2010)), ]),
                                partial_type=args.partial_type,
                                partial_rate=args.flipping_rate
                                )
        test_dataset = cifar10(root='./datasets/cifar10/',
                               download=True,
                               train=False,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                  (0.2023, 0.1994, 0.2010)), ]),
                               partial_type=args.partial_type,
                               partial_rate=args.flipping_rate
                               )
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, drop_last=False, shuffle=True)
        train_loader_temp = DataLoader(dataset=train_dataset_temp, batch_size=args.batch_size,
                                       num_workers=args.num_workers, drop_last=False, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False)
        for i in range(len(train_dataset)):
            for j in range(args.classes):
                if train_loader_temp.dataset.train_final_labels[i, j] > 0:
                    train_loader_temp.dataset.train_final_labels[i, j] = 1
        model = ResNet34(num_classes=args.classes).to(args.device)

    # warm up model to distill examples
    best_acc = 0.
    cudnn.benchmark = True
    optimizer_warmup = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print('==> Warm up model..')
    for epoch in range(args.warmup_epochs):
        model.train()
        for i, (images, labels, trues, indexes) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            output = model(images)

            loss, new_label = partial_loss(output, labels)
            optimizer_warmup.zero_grad()
            loss.backward()
            optimizer_warmup.step()

            # update weights
            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()

        test_acc = evaluate(args, test_loader, model)
        print('Warm up Epoch [%d / %d] Test Accuracy on the %s test data: Model1 %.2f %%' % (
            epoch + 1, args.warmup_epochs, len(test_dataset), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'warmup_model.pth'))

    # distill examples
    threshold = (1 + args.rho) / 2
    model.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location=args.device))
    test_acc = evaluate(args, test_loader, model)
    print('Loading Test Accuracy on the %s val data: Model1 %.2f %%' % (
        len(test_dataset), test_acc))

    distilled_example_indexes = []
    distilled_example_labels = []
    print('==> Distill examples..')
    model.eval()
    for i, (images, labels, trues, indexes) in enumerate(train_loader_temp):
        images = images.to(args.device)
        logits = F.softmax(model(images), dim=1)
        logits_max = torch.max(logits, dim=1)
        mask = logits_max[0] > threshold
        distilled_example_indexes.extend(indexes[mask])
        distilled_example_labels.extend(logits_max[1].cpu()[mask])

    distilled_example_indexes = np.array(distilled_example_indexes)
    distilled_bayes_labels = np.array(distilled_example_labels)
    distilled_images, distilled_partial_labels, distilled_true_labels = train_dataset_temp.train_data[distilled_example_indexes], \
                                                                        train_dataset_temp.train_final_labels[distilled_example_indexes], \
                                                                        train_dataset_temp.train_labels[distilled_example_indexes]
    print('==> Distilling finished..')

    print('Number of distilled examples: %d' % len(distilled_example_indexes))
    print('Accuracy of distilled examples collection: %.2f %%' % ((distilled_bayes_labels == np.array(distilled_true_labels)).sum() * 100 / len(distilled_bayes_labels)))
    np.save(os.path.join(model_dir, 'distilled_images.npy'), distilled_images)
    np.save(os.path.join(model_dir, 'distilled_bayes_labels.npy'), distilled_bayes_labels)
    np.save(os.path.join(model_dir, 'distilled_partial_labels.npy'), distilled_partial_labels)
    np.save(os.path.join(model_dir, 'distilled_true_labels.npy'), distilled_true_labels)

    print('==> Distilled dataset building..')
    distilled_images = np.load(os.path.join(model_dir, 'distilled_images.npy'))
    distilled_bayes_labels = np.load(os.path.join(model_dir, 'distilled_bayes_labels.npy'))
    distilled_partial_labels = np.load(os.path.join(model_dir, 'distilled_partial_labels.npy'))

    if args.dataset == 'mnist':
        distilled_dataset_ = datasets.distilled_datasets.distilled_dataset(distilled_images,
                                                                           distilled_partial_labels,
                                                                           distilled_bayes_labels,
                                                                           transform=transforms.Compose([
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,),
                                                                                                    (0.3081,)), ]),
                                                                           target_transform=transform_target)

        Bayesian_T_Network = Bayes_ResNet18_M(100)
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location=args.device))
        for name, parameter in model.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    elif args.dataset == 'fashion_mnist':
        distilled_dataset_ = datasets.distilled_datasets.distilled_dataset(distilled_images,
                                                                           distilled_partial_labels,
                                                                           distilled_bayes_labels,
                                                                           transform=transforms.Compose([
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,),
                                                                                                    (0.3081,)), ]),
                                                                           target_transform=transform_target)

        Bayesian_T_Network = Bayes_ResNet18_M(100)
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location=args.device))
        for name, parameter in model.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    elif args.dataset == 'cifar10':
        distilled_dataset_ = datasets.distilled_datasets.distilled_dataset(distilled_images,
                                                    distilled_partial_labels,
                                                    distilled_bayes_labels,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                             (0.2023, 0.1994, 0.2010)), ]),
                                                    target_transform=transform_target
                                                    )
        Bayesian_T_Network = Bayes_ResNet34(100)
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location=args.device))
        for name, parameter in model.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)


    train_loader_distilled = DataLoader(dataset=distilled_dataset_, batch_size=args.batch_size,
                                        num_workers=args.num_workers, drop_last=False, shuffle=True)

    print('==> Bayesian Transition Network training..')
    Bayesian_T_Network.to(args.device)
    optimizer_bayes = torch.optim.SGD(Bayesian_T_Network.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_function = LSEP()
    for epoch in range(args.bayes_epochs):
        bayes_loss = 0.
        total = 0
        Bayesian_T_Network.train()
        for i, (images, bayes_labels, partial_labels, indexes) in enumerate(train_loader_distilled):
            images = images.to(args.device)
            bayes_labels = bayes_labels.to(args.device)
            partial_labels = partial_labels.to(args.device)
            batch_matrix = Bayesian_T_Network(images)
            loss = loss_function(batch_matrix, bayes_labels, partial_labels)
            optimizer_bayes.zero_grad()
            loss.backward()
            optimizer_bayes.step()
            bayes_loss += loss.item()
            total += 1

        print('Bayesian-T Training Epoch [%d / %d], Loss: %.4f' % (epoch + 1, args.epochs, bayes_loss / total))
        torch.save(Bayesian_T_Network.state_dict(), os.path.join(model_dir, 'BayesianT.pth'))

    print('==> Bayesian Transition Network training finished..')

    test_acc_list = []

    model.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location=args.device))
    nn.init.constant_(model.T_revision.weight, 0.0)

    Bayesian_T_Network.load_state_dict(torch.load(os.path.join(model_dir, 'BayesianT.pth'), map_location=args.device))
    print('Loading Test Accuracy on the %s test data: Model1 %.2f %%' % (
        len(test_dataset), evaluate(args, test_loader, model)))
    optimizer_r = torch.optim.Adam(model.parameters(), lr=5e-7, weight_decay=1e-4)
    for i in range(len(train_dataset)):
        for j in range(args.classes):
            if train_loader.dataset.train_final_labels[i, j] > 0:
                train_loader.dataset.train_final_labels[i, j] = 1

    print('==> Bayesian Revision Network training..')
    for epoch in range(args.epochs):
        model.train()
        Bayesian_T_Network.eval()
        train_total = 0
        train_correct = 0
        train_loss = 0.

        for i, (images, labels, trues, indexes) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            trues = trues.to(args.device)
            optimizer_r.zero_grad()
            output, delta = model(images, revision=args.revision)
            bayes_post = F.softmax(output, dim=1)
            bayes_labels = torch.argmax(bayes_post, dim=1)

            delta = delta.repeat(len(labels), 1, 1)
            T = Bayesian_T_Network(images)
            if args.revision:
                T = norm(T + delta)
            loss = loss_function(T, bayes_labels, labels)

            prec, = accuracy(bayes_post, trues, topk=(1,))
            train_total += 1
            train_correct += prec
            train_loss += loss.item()
            loss.backward()
            optimizer_r.step()

        train_acc = float(train_correct) / float(train_total)
        test_acc = evaluate(args, test_loader, model)
        test_acc_list.append(test_acc)
        print('Epoch [%d / %d] Train Accuracy on the %s train data: Model1 %.4f %%, loss: %.4f ' % (
            epoch + 1, args.epochs, len(train_dataset), train_acc, train_loss / train_total))
        print('Epoch [%d / %d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
            epoch + 1, args.epochs, len(test_dataset), test_acc))

    id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[id]
    print('Max Test Accuracy on the %s test data: Model1 %.4f %%' % (len(test_dataset), test_acc_max))

    return test_acc_max


if __name__ == '__main__':
    args.result_dir = os.path.join(args.result_dir, args.dataset)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if args.print:
        file = open(os.path.join(args.result_dir, f'flipping_rate:{args.flipping_rate}_rho:{args.rho}.txt'), 'a')
        sys.stdout = file
        sys.stderr = file
    acc = main(args)
    os.system('shutdown')