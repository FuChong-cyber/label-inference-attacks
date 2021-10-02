import argparse
import ast
import os
import time
import dill
from time import time
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import datasets.get_dataset as get_dataset
from my_utils import utils
from models import model_sets, bottom_model_plus
import my_optimizers
import possible_defenses
import torch.nn.functional as F

plt.switch_backend('agg')

D_ = 2 ** 13
BATCH_SIZE = 1000


def split_data_xa(data):
    if args.dataset_name == 'Liver':
        x_a = data[:, 0:args.half]
    elif args.dataset_name in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
    elif args.dataset_name == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
    elif args.dataset_name == 'Criteo':
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    return x_a


def create_model(size_bottom_out=10, num_classes=10):
    model = bottom_model_plus.BottomModelPlus(size_bottom_out, num_classes)
    model = model.cuda()
    return model


def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def train_model(train_loader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = split_data_xa(data)
        data = data.float().cuda()
        target = target.long().cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        interval = int(0.1 * len(train_loader))
        interval_num = int(batch_idx / interval)
        interval_left = batch_idx % interval
        if interval_left == 0:
            print(f"{interval_num}0% completed, loss is {loss}")


def test_per_epoch(test_loader, model, k=5):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().cuda()
            target = target.long().cuda()
            # set all sub-models to eval mode.
            model.eval()
            # run forward process of the whole framework
            x_a = split_data_xa(data)
            output = model(x_a)
            correct_top1_batch, correct_topk_batch = correct_counter(output, target, (1, k))
            # sum up batch loss
            test_loss += F.cross_entropy(output, target).data.item()
            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch
            # print("one batch done")
            count += 1
            if count % int(0.1 * len(test_loader)) == 0 and count // int(0.1 * len(test_loader)) > 0:
                print(f'{count // int(0.1 * len(test_loader))}0 % completed...')
            if args.dataset_name == 'Criteo' and count == test_loader.train_batches_num:
                break
        if args.dataset_name == 'Criteo':
            num_samples = len(test_loader) * BATCH_SIZE
        else:
            num_samples = len(test_loader.dataset)
        test_loss /= num_samples
        print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples,
            k,
            correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples
        ))


def set_loaders():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)
    if args.dataset_name == 'Criteo':
        train_loader = train_dataset
        test_loader = test_dataset
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, shuffle=True,
            batch_size=args.batch_size,
            # num_workers=args.workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            # num_workers=args.workers
        )
    return train_loader, test_loader


def main():
    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    setting_str += "_upperbound"
    setting_str += "_"
    setting_str += "half="
    setting_str += str(args.half)
    print("settings:", setting_str)

    print('==> Preparing {}'.format(args.dataset_name))
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    size_bottom_out = dataset_setup.size_bottom_out
    num_classes = dataset_setup.num_classes

    model = create_model(size_bottom_out=size_bottom_out, num_classes=num_classes)
    model.bottom_model = model_sets.BottomModel(args.dataset_name).get_model(args.half, True)
    model = model.cuda()
    cudnn.benchmark = True

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[stone1, stone2], gamma=args.step_gamma)

    train_loader, val_loader = set_loaders()

    dir_save_model = args.save_dir + f"/saved_models/{args.dataset_name}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    # start training. do evaluation every epoch.
    print('Test the initialized model:')
    print('Evaluation on the training dataset:')
    # test_per_epoch(test_loader=train_loader, model=model, k=args.k)
    print('Evaluation on the testing dataset:')
    # test_per_epoch(test_loader=val_loader, model=model, k=args.k)
    for epoch in range(args.epochs):
        print('optimizer current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_model(train_loader, model, optimizer)
        lr_scheduler.step()

        if epoch == args.epochs - 1:
            txt_name = f"{args.dataset_name}_saved_framework{setting_str}"
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Evaluation on the training dataset:')
                test_per_epoch(test_loader=train_loader, model=model, k=args.k)
                print('Evaluation on the testing dataset:')
                test_per_epoch(test_loader=val_loader, model=model, k=args.k)
                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

        print('Evaluation on the training dataset:')
        test_per_epoch(test_loader=train_loader, model=model, k=args.k)
        print('Evaluation on the testing dataset:')
        test_per_epoch(test_loader=val_loader, model=model, k=args.k)

    # save model
    torch.save(model, os.path.join(dir_save_model, f"{args.dataset_name}_saved_framework{setting_str}.pth"),
               pickle_module=dill)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('-d', '--dataset-name', default='Criteo', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Liver', 'Criteo'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='D:/Datasets/Criteo/criteo.csv')
    '''
    'D:/Datasets/CIFAR10'
    'D:/Datasets/CIFAR100'
    'D:/Datasets/TinyImageNet'
    'D:/Datasets/CINIC10L'
    'D:/Datasets/BC_IDC'
    'D:/Datasets/Criteo/criteo1e?.csv'
    '''
    # framework paras
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    # saving path paras
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='D:/MyCodes/label_inference_attacks_against_vfl/saved_experiment_results', type=str)
    # training paras
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of datasets loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')  # TinyImageNet=5e-2
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--step-gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    args = parser.parse_args()
    main()
