import argparse
import copy
import os
import math
import sys

import dill
import torch.nn as nn
import torch.nn.functional as F

from models.read_data_text import *
from models.mixtext import MixText
from models.bottom_model_plus import BottomModelPlus
from vfl_framework import VflFramework

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=4, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=5e-6, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=10,
                    help='number of labeled data per class')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=1000,
                    help='number of labeled data')


parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='D:/Datasets/yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[7, 9, 12], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=16, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')

# checkpoints paras (used for trained bottom model in our attack)
parser.add_argument('--resume-dir',
                    default='D:/MyCodes/label_inference_attacks_against_vfl/saved_experiment_results'
                            '/saved_models/',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint', )
parser.add_argument('--resume-name',
                    default='Yahoo_saved_framework_lr=0.001_normal_.pth',
                    type=str, metavar='NAME',
                    help='file name of the latest checkpoint', )
parser.add_argument('--dataset-name', default="Yahoo", type=str)

# saving path paras
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models and csv files',
                    default='D:/MyCodes/label_inference_attacks_against_vfl/saved_experiment_results', type=str)

args = parser.parse_args()
args.resume = args.resume_dir + f'{args.dataset_name}_saved_models/' + args.resume_name

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)


def main():
    dir_save_model = args.save_dir + f"/saved_models/{args.dataset_name}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    # Define the model, set the optimizer
    model = BottomModelPlus(size_bottom_out=10, num_classes=10)
    model = model.cuda()

    model = MixText(n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, pickle_module=dill)

        print("checkpoint:", checkpoint.malicious_bottom_model_a)
        model.bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)
        # model.fc.apply(weights_init_ones)
        # ema_model.fc.apply(weights_init_ones)

    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration

    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    test_accs = []

    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        #print("epoch {}, train acc {}".format(epoch, train_acc))

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

        if epoch == args.epochs - 1:
            txt_name = 'model_completion_' + args.resume_name
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Epoch: ', epoch)
                print('Best acc:')
                print(best_acc)
                print('Test acc:')
                print(test_accs)
                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1

        if not train_aug:
            try:
                data_zip, target_length_zip = labeled_train_iter.next()
                inputs_x, _ = data_zip
                targets_x, inputs_x_length, _ = target_length_zip
            except:
                labeled_train_iter = iter(labeled_trainloader)
                data_zip, target_length_zip = labeled_train_iter.next()
                inputs_x, _ = data_zip
                targets_x, inputs_x_length, _ = target_length_zip
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
        try:
            ((inputs_u, inputs_u2,  inputs_ori),
             (length_u, length_u2,  length_ori)), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            ((inputs_u, inputs_u2, inputs_ori),
             (length_u, length_u2, length_ori)), _ = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1).long(), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            outputs_ori = model(inputs_ori)

            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (1)
            # Do a sharpen here.
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1

        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1-l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            logits = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1-l))
                    if length1 + length2 > 256:
                        length2 = 256-length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits = model(mixed_input, sent_size=512)

                #mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b)/2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)

        if mix_ == 1:
            loss = Lx + w * Lu
        else:
            loss = Lx + w * Lu + w2 * Lu2

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, zip_ in enumerate(valloader):
            (inputs, _), (targets, length, _) = zip_
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labels and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
