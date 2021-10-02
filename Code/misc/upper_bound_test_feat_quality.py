import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import MixMatch_pytorch.models.bottom_model_plus as models
from 即将删除的文件.liver_models import BottomModel
import numpy as np
import os

parser = argparse.ArgumentParser(description='liver framework settings')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--use-cuda', default=True, type=bool, help='use cuda device or not')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-model-dir', dest='save_model_dir',
                    help='The directory used to save the trained models',
                    default='./saved_models/liver_saved_models', type=str)
parser.add_argument('--save-csv-dir', dest='save_csv_dir',
                    help='The directory used to save the csv files(outputs a tsne embedding)',
                    default='./csv_files/liver_csv_files', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=50)
parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA',
                    type=bool, default=True)
parser.add_argument('--path-dataset', help='path_dataset',
                    type=str, default='./datasets/HCV-Egy-Data/HCV-Egy-Data.csv')
parser.add_argument('--use-mal-optim', help='use mal optim or not',
                    type=bool, default=False)
parser.add_argument('--half', help='number of features owned by                                    the adversarial participant (int)',
                    type=int, default=11)

# 4 possible defenses on/off
parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                    type=bool, default=False)
parser.add_argument('--gc', help='turn_on_gradient_compression',
                    type=bool, default=False)
parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                    type=bool, default=False)
parser.add_argument('--sign-sgd', help='turn_on_sign_sgd',
                    type=bool, default=False)
# setting about possible defenses
parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                    type=float, default=0.75)
parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                    type=float, default=0.75)
parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                    type=float, default=1e-1)


def train_per_batch(x_a, target, model, optimizer, loss=nn.CrossEntropyLoss()):
    optimizer.zero_grad()
    output = model(x_a)
    loss_value = loss(output, target)
    loss_value.backward()
    optimizer.step()
    return loss_value


def test_per_epoch(test_loader, model, loss=nn.CrossEntropyLoss()):
    test_loss = 0
    correct = 0

    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    # set all sub-models to eval mode.
    model.eval()

    for data, target in test_loader:
        data = data.float()
        target = target.long()
        if args.use_cuda:
            data = data.cuda()
            target = target.cuda()
        # run forward process of the whole framework
        x_a = data[:, :args.half]  # (batch size，features)
        output = model(x_a)

        # sum up batch loss
        test_loss += loss(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # print(pred)
        target_data = target.data.view_as(pred)
        # print("target_data:", target_data)
        # total number of correctly predicted samples
        correct += pred.eq(target_data).cpu().sum()
        # number of correctly predicted positive samples
        if args.use_cuda:
            target_data = target_data.cpu()
            pred = pred.cpu()
        y_true = np.array(target_data)
        y_pred = np.array(pred)
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1.:
                    TP_samples_num += 1
                else:
                    TN_samples_num += 1
                right_samples_num += 1
            else:
                if y_pred[i] == 1.:
                    FP_samples_num += 1
                else:
                    FN_samples_num += 1
                wrong_samples_num += 1

    total_samples_num = right_samples_num + wrong_samples_num
    print('Accuracy: ', right_samples_num / total_samples_num)
    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    print('Precision:', precision, end='')
    print('  Recall:', recall, end='')
    print("  F1", f1)
    print('TP Samples Num: ', TP_samples_num)
    print('TN Samples Num: ', TN_samples_num)
    print('FP Samples Num: ', FP_samples_num)
    print('FN Samples Num: ', FN_samples_num)
    print('Right Samples Num: ', right_samples_num)
    print('Wrong Samples Num: ', wrong_samples_num)
    print('Total Samples Num: ', total_samples_num)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        float(correct) * 100 / len(test_loader.dataset)))
    model.train()


def main():
    global args
    args = parser.parse_args()

    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    if args.use_mal_optim:
        setting_str += "_"
        setting_str += "mal"
    else:
        setting_str += "_"
        setting_str += "normal"
    if args.ppdl:
        setting_str += "_"
        setting_str += "ppdl-theta_u="
        setting_str += str(args.ppdl_theta_u)
    if args.gc:
        setting_str += "_"
        setting_str += "gc-preserved_ percent="
        setting_str += str(args.gc_preserved_percent)
    if args.lap_noise:
        setting_str += "_"
        setting_str += "lap_noise-scale="
        setting_str += str(args.noise_scale)
    setting_str += "_"
    setting_str += "num_adv_feat="
    setting_str += str(args.half)
    print("settings:", setting_str)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_csv_dir):
        os.makedirs(args.save_csv_dir)

    model = models.BottomModelPlus(4, 4)
    model.bottom_model = BottomModel(half=args.half, is_adversary=True)
    if args.use_cuda:
        model = model.cuda()

    # liver Dataset

    train_loader = liver_helper.LiverDataset(args.path_dataset, args.batch_size).get_trainloader()
    test_loader = liver_helper.LiverDataset(args.path_dataset, args.batch_size).get_testloader()
    # Data Loader (Input Pipeline)
    print("len train_loader:", len(train_loader))
    print("len test_loader:", len(test_loader))
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader): 
            # print("batch_idx:", batch_idx)
            data = data.float()
            target = target.long()
            if args.use_cuda:
                data = data.cuda()
                target = target.cuda()
            x_a = data[:, :args.half]
            # print("datasets:", datasets)
            loss = train_per_batch(x_a, target, model, optimizer)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
                with torch.no_grad():
                    test_per_epoch(test_loader=test_loader, model=model)
        with torch.no_grad():
            test_per_epoch(test_loader=test_loader, model=model)


if __name__ == '__main__':
    main()
