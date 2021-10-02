import argparse
import ast
import os
import sys

import dill
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from my_utils import utils
from models import idc_models
import my_optimizers
import datasets.get_dataset as get_dataset
import possible_defenses

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='ResNets for BreastCancer subtype classification(IDC) in pytorch')
# dataset paras
parser.add_argument('--path-dataset', help='path_dataset',
                    type=str, default='D:/Datasets/BC_IDC')
# saving path paras
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models and csv files',
                    default='D:/MyCodes/label_inference_attacks_against_vfl/saved_experiment_results', type=str)
# framework paras
parser.add_argument('--party-num', help='party-num',
                    type=int, default=2)
parser.add_argument('--overlap',
                    help='whether the attacker uses more features',
                    type=ast.literal_eval, default=False)
parser.add_argument('--use-top-model',
                    help='whether the vfl framework has a top model. If set to False, the program will automatically'
                         'evaluate the direct label inference attack.',
                    type=ast.literal_eval, default=True)
# attack paras
parser.add_argument('--use-mal-optim',
                    help='whether the attacker uses the malicious optimizer',
                    type=ast.literal_eval, default=True)
parser.add_argument('--use-mal-optim-all',
                    help='whether all participants use the malicious optimizer. If set to '
                         'True, use_mal_optim will be automatically set to True.',
                    type=ast.literal_eval, default=False)
parser.add_argument('--use-mal-optim-top',
                    help='whether the server(top model) uses the malicious optimizer',
                    type=ast.literal_eval, default=False)
# visualization paras
parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA',
                    type=ast.literal_eval, default=False)
# possible defenses paras
parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                    type=ast.literal_eval, default=False)
parser.add_argument('--gc', help='turn_on_gradient_compression',
                    type=ast.literal_eval, default=False)
parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                    type=ast.literal_eval, default=False)
parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
parser.add_argument('--sign-sgd', help='turn_on_sign_sgd',
                    type=ast.literal_eval, default=False)
# setting about possible defenses
parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                    type=float, default=0.25)
parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                    type=float, default=0.1)
parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                    type=float, default=1e-1)
parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)
# training paras
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of datasets loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
best_prec1 = 0


class IdcVflFramework(nn.Module):

    def __init__(self, ppdl, gc, lap_noise, ss):
        super(IdcVflFramework, self).__init__()
        # counter for direct label inference attack
        self.inferred_correct = 0
        self.inferred_wrong = 0
        self.direct_attack_on = False
        # bottom model a can collect output_a for label inference attack
        self.collect_outputs_a = False
        self.outputs_a = torch.tensor([]).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.if_collect_training_dataset_labels = False

        # adversarial options
        self.defense_ppdl = ppdl
        self.defense_gc = gc
        self.defense_lap_noise = lap_noise
        self.defense_ss = ss
        self.defense_multistep_grad = args.multistep_grad

        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1.0])).cuda()
        self.loss_func_bottom_model = utils.keep_predict_loss

        # top model
        # By default, each bottom model has a 5-dim output
        self.top_model = idc_models.TopModel(dims_in=5 * args.party_num)
        # bottom models as a list. self.bottom_models[0] as the malicious party.
        self.bottom_models = []
        for bottom_model_id in range(args.party_num):
            if args.use_top_model:
                self.bottom_models.append(idc_models.BottomModel().cuda())
            else:
                self.bottom_models.append(idc_models.BottomModelForDirect().cuda())

        # overlap features test
        if args.overlap:
            self.bottom_models[0] = idc_models.BottomModelOverlap().cuda()

        # bottom model optimizers as a list.
        self.bottom_model_optimizers = []

        # This setting is for adversarial experiments except sign SGD
        if args.use_mal_optim_top:
            self.optimizer_top_model = my_optimizers.MaliciousSGD(self.top_model.parameters(),
                                                                  lr=args.lr,
                                                                  momentum=args.momentum,
                                                                  weight_decay=args.weight_decay)
        else:
            self.optimizer_top_model = optim.SGD(self.top_model.parameters(),
                                                 lr=args.lr,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
        if args.use_mal_optim:
            self.bottom_model_optimizers.append(
                my_optimizers.MaliciousSGD(
                    self.bottom_models[0].parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            )
        else:
            self.bottom_model_optimizers.append(
                optim.SGD(
                    self.bottom_models[0].parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            )
        if args.use_mal_optim_all:
            for i in range(1, args.party_num):
                self.bottom_model_optimizers.append(
                    my_optimizers.MaliciousSGD(
                        self.bottom_models[i].parameters(),
                        lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
                )
        else:
            for i in range(1, args.party_num):
                self.bottom_model_optimizers.append(
                    optim.SGD(
                        self.bottom_models[i].parameters(),
                        lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
                )

    def forward(self, x):

        # in vertical federated setting, each party has non-lapping features of the same sample
        input_images_list = []
        for i in range(args.party_num):
            input_images_list.append(x[:, i:i + 1, :, :, :].squeeze(1))
        bottom_model_outputs_list = []
        for i in range(args.party_num):
            # overlap features test
            if i == 0 and args.overlap:
                bottom_model_outputs_list.append(self.bottom_models[i](
                    torch.cat((input_images_list[0], input_images_list[1], input_images_list[2]), dim=3)))
            else:
                bottom_model_outputs_list.append(self.bottom_models[i](input_images_list[i]))

        if not args.use_top_model:
            out = None
            for i in range(args.party_num):
                if out is None:
                    out = bottom_model_outputs_list[i]
                else:
                    out += bottom_model_outputs_list[i]
        else:
            bottom_model_output_all = torch.stack(bottom_model_outputs_list)
            out = self.top_model(bottom_model_output_all)
        return out

    def simulate_train_round_per_batch(self, data, target):
        timer_mal = 0
        timer_benign = 0
        # simulate: bottom models forward, top model forward, top model backward and update, bottom backward and update

        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        if self.if_collect_training_dataset_labels:
            self.labels_training_dataset = torch.cat((self.labels_training_dataset, target), dim=0)
        # store grad of input of top model/outputs of bottom models
        input_tensors_top_model = []
        for i in range(args.party_num):
            input_tensors_top_model.append(torch.tensor([], requires_grad=False))
        output_tensors_bottom_model = []

        # --bottom models forward--
        input_images_list = []
        for i in range(args.party_num):
            input_images_list.append(data[:, i:i + 1, :, :, :].squeeze(1))
        for i in range(args.party_num):
            self.bottom_models[i].train(mode=True)
            # overlap features test
            if i == 0 and args.overlap:
                output_tensors_bottom_model.append(self.bottom_models[i](
                    torch.cat((input_images_list[0], input_images_list[1], input_images_list[2]), dim=3)))
            else:
                output_tensors_bottom_model.append(self.bottom_models[i](input_images_list[i]))
            if i == 0:
                # bottom model a can collect output_a for label inference attack
                if self.collect_outputs_a:
                    self.outputs_a = torch.cat((self.outputs_a, output_tensors_bottom_model[0].data))
            input_tensors_top_model[i].data = output_tensors_bottom_model[i].data

        grads_output_bottom_model_list = []
        if args.use_top_model:
            bottom_model_output_all = torch.tensor([]).cuda()
            for i in range(args.party_num):
                bottom_model_output_all = torch.cat((bottom_model_output_all, input_tensors_top_model[i]), dim=1)
            # bottom_model_output_all = torch.stack(input_tensors_top_model)
            bottom_model_output_all.requires_grad = True
            # -top model forward-
            self.top_model.train(mode=True)
            output_framework = self.top_model(bottom_model_output_all)
            # --top model backward/update--
            # top model loss input tensor
            loss_framework = idc_models.update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                                   model=self.top_model,
                                                                   output=output_framework,
                                                                   batch_target=target,
                                                                   loss_func=self.loss_func_top_model)
            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's
            # target
            for i in range(args.party_num):
                grads_output_bottom_model_list.append(bottom_model_output_all.grad[:, i * 5:(i + 1) * 5])
        else:
            for i in range(args.party_num):
                input_tensors_top_model[i] = input_tensors_top_model[i].cuda()
                input_tensors_top_model[i].requires_grad = True
            output_framework = torch.zeros_like(input_tensors_top_model[0])
            output_framework = output_framework.cuda()
            # output_framework.require
            for i in range(args.party_num):
                output_framework += input_tensors_top_model[i]
            loss_framework = self.loss_func_top_model(output_framework, target)
            loss_framework.backward()
            for i in range(args.party_num):
                grads_output_bottom_model_list.append(input_tensors_top_model[i].grad)

        # defenses here: the server(who controls top model) can defend against label inference attack by protecting
        # print("before defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)
        # gradients sent to bottom models
        model_all_layers_grads_list = grads_output_bottom_model_list
        # privacy preserving deep learning
        if self.defense_ppdl:
            for tensor_id in range(len(model_all_layers_grads_list)):
                possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1,
                                             layer_grad_list=[model_all_layers_grads_list[tensor_id]],
                                             theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
        # gradient compression
        if self.defense_gc:
            tensor_pruner = possible_defenses.TensorPruner(zip_percent=args.gc_preserved_percent)
            for tensor_id in range(len(model_all_layers_grads_list)):
                tensor_pruner.update_thresh_hold(model_all_layers_grads_list[tensor_id])
                # print("tensor_pruner.thresh_hold:", tensor_pruner.thresh_hold)
                model_all_layers_grads_list[tensor_id] = tensor_pruner.prune_tensor(
                    model_all_layers_grads_list[tensor_id])
        # differential privacy
        if self.defense_lap_noise:
            dp = possible_defenses.DPLaplacianNoiseApplyer(beta=args.noise_scale)
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = dp.laplace_mech(model_all_layers_grads_list[tensor_id])
        # multistep gradient
        if self.defense_multistep_grad:
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = possible_defenses.multistep_gradient(
                    model_all_layers_grads_list[tensor_id], bins_num=args.multistep_grad_bins,
                    bound_abs=args.multistep_grad_bound_abs)
        # print("after defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)

        # server sends back output_tensor_server_a.grad to the adversary(participant a), so
        # the adversary can use this gradient to perform direct label inference attack.
        if self.direct_attack_on:
            for sample_id in range(len(model_all_layers_grads_list[0])):
                grad_per_sample = model_all_layers_grads_list[0][sample_id]
                for logit_id in range(len(grad_per_sample)):
                    if grad_per_sample[logit_id] < 0:
                        inferred_label = logit_id
                        if inferred_label == target[sample_id]:
                            self.inferred_correct += 1
                        else:
                            self.inferred_wrong += 1
                        break

        # --bottom models backward/update--
        # -bottom model 0: backward/update-
        # print("malicious_bottom_model 0")
        idc_models.update_bottom_model_one_batch(optimizer=self.bottom_model_optimizers[0],
                                                 model=self.bottom_models[0],
                                                 output=output_tensors_bottom_model[0],
                                                 batch_target=grads_output_bottom_model_list[0],
                                                 loss_func=self.loss_func_bottom_model)
        # -benign bottom models: backward/update-
        # print("benign_bottom_models")
        for i in range(1, args.party_num):
            idc_models.update_bottom_model_one_batch(optimizer=self.bottom_model_optimizers[i],
                                                     model=self.bottom_models[i],
                                                     output=output_tensors_bottom_model[i],
                                                     batch_target=grads_output_bottom_model_list[i],
                                                     loss_func=self.loss_func_bottom_model)
        return loss_framework


def test_per_epoch(test_loader, framework, criterion):
    test_loss = 0
    correct = 0

    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().cuda()
            target = target.long().cuda()
            # set all sub-models to eval mode.
            for i in range(args.party_num):
                framework.bottom_models[i].eval()
            framework.top_model.eval()
            # run forward process of the whole framework
            output_tensors_bottom_models = torch.tensor([]).cuda()
            for i in range(args.party_num):
                input_images = data[:, i:i + 1, :, :, :].squeeze(1)
                # overlap test
                if i == 0 and args.overlap:
                    output_tensors_bottom_models = torch.cat((output_tensors_bottom_models,
                                                              framework.bottom_models[i](torch.cat(
                                                                  (data[:, i:i + 1, :, :, :],
                                                                   data[:, i + 1:i + 2, :, :, :],
                                                                   data[:, i + 2:i + 3, :, :, :],)
                                                                  , dim=4).squeeze(1))),
                                                             dim=1)
                elif args.use_top_model:
                    output_tensors_bottom_models = torch.cat((output_tensors_bottom_models,
                                                              framework.bottom_models[i](input_images)),
                                                             dim=1)
                else:
                    if len(output_tensors_bottom_models.shape) == 1:
                        output_tensors_bottom_models = framework.bottom_models[i](input_images)
                    else:
                        output_tensors_bottom_models += framework.bottom_models[i](input_images)
            if args.use_top_model:
                output_framework = framework.top_model(output_tensors_bottom_models)
            else:
                output_framework = output_tensors_bottom_models

            # sum up batch loss
            test_loss += criterion(output_framework, target).data.item()
            # get the index of the max log-probability
            pred = output_framework.data.max(1, keepdim=True)[1]
            # print(pred)
            target_data = target.data.view_as(pred)
            # print("target_data:", target_data)
            correct += pred.eq(target_data).cpu().sum()

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
        print('\nLoss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.00 * correct / len(test_loader.dataset)))


def main():
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.0])).cuda()

    global args, best_prec1
    args = parser.parse_args()
    if args.use_mal_optim_all:
        args.use_mal_optim = True
    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    if args.use_mal_optim:
        setting_str += "_"
        setting_str += "mal"
        if args.use_mal_optim_all:
            setting_str += "-all"
        if args.use_mal_optim_top:
            setting_str += "-top"
    else:
        setting_str += "_"
        setting_str += "normal"
    if args.ppdl:
        setting_str += "_"
        setting_str += "ppdl-theta_u="
        setting_str += str(args.ppdl_theta_u)
    if args.gc:
        setting_str += "_"
        setting_str += "gc-preserved_percent="
        setting_str += str(args.gc_preserved_percent)
    if args.lap_noise:
        setting_str += "_"
        setting_str += "lap_noise-scale="
        setting_str += str(args.noise_scale)
    if args.multistep_grad:
        setting_str += "_"
        setting_str += "multistep_grad_bins="
        setting_str += str(args.multistep_grad_bins)
    setting_str += "_"
    setting_str += "party-num="
    setting_str += str(args.party_num)
    print("settings:", setting_str)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = IdcVflFramework(ppdl=args.ppdl,
                            gc=args.gc,
                            lap_noise=args.lap_noise,
                            ss=args.sign_sgd)
    model = model.cuda()
    cudnn.benchmark = True
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_top_model,
                                                                  milestones=[4, 6],
                                                                  last_epoch=-1)
    lr_scheduler_bottom_models_list = []
    for i in range(args.party_num):
        lr_scheduler_bottom_models_list.append(
            torch.optim.lr_scheduler.MultiStepLR(model.bottom_model_optimizers[i],
                                                 milestones=[4, 6], last_epoch=-1)
        )
    args.dataset_name = 'BC_IDC'
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, args.party_num, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, args.party_num, False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size * 15,
        num_workers=args.workers
    )

    dir_save_model = args.save_dir + "/saved_models/BC_IDC_saved_models/"
    name_save_model = "BC_IDC_saved_framework" + setting_str + ".pth"

    # start training. do evaluation every epoch.
    print('Test the initialized model:')
    print('Evaluation on the training dataset:')
    test_per_epoch(test_loader=train_loader, framework=model, criterion=criterion)
    print('Evaluation on the testing dataset:')
    test_per_epoch(test_loader=test_loader, framework=model, criterion=criterion)
    for epoch in range(args.epochs):

        print('model.optimizer_top_model current lr {:.5e}'.format(model.optimizer_top_model.param_groups[0]['lr']))

        if epoch == 0:
            model.direct_attack_on = True
        else:
            model.direct_attack_on = False

        if epoch == args.epochs - 1:
            model.collect_outputs_a = True
            model.if_collect_training_dataset_labels = True
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().cuda()
            target = target.long().cuda()
            loss_framework = model.simulate_train_round_per_batch(data, target)
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_framework.data.item()))
        lr_scheduler_top_model.step()
        for i in range(args.party_num):
            lr_scheduler_bottom_models_list[i].step()

        if epoch == args.epochs - 1:
            txt_name = f"idc_saved_framework{setting_str}"
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Evaluation on the training dataset:')
                test_per_epoch(test_loader=train_loader, framework=model, criterion=criterion)
                print('Evaluation on the testing dataset:')
                test_per_epoch(test_loader=test_loader, framework=model, criterion=criterion)
                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

        print('Evaluation on the training dataset:')
        test_per_epoch(test_loader=train_loader, framework=model, criterion=criterion)
        print('Evaluation on the testing dataset:')
        test_per_epoch(test_loader=test_loader, framework=model, criterion=criterion)

        if not args.use_top_model:
            # performance of the direct label inference attack
            print("inferred correctly:", model.inferred_correct)
            num_all_train_samples = len(train_loader.dataset)
            print("all:", num_all_train_samples)
            print("Direct label inference accuracy:", model.inferred_correct / num_all_train_samples)
            print("Direct label inference attack evaluated. Existing...")
            exit()

    # save model
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)
    torch.save(model, dir_save_model + name_save_model, pickle_module=dill)

    if args.if_cluster_outputsA:
        outputsA_list = model.outputs_a.detach().clone().cpu().numpy().tolist()
        labels_list = model.labels_training_dataset.detach().clone().cpu().numpy().tolist()
        # plot TSNE cluster result
        outputsA_pca_tsne = TSNE()
        outputsA_pca_tsne.fit_transform(outputsA_list)
        df_outputsA_pca_tsne = pd.DataFrame(outputsA_pca_tsne.embedding_, index=labels_list)
        dir_save_csv = args.save_dir + "/csv_files/BC_IDC_csv_files/"
        name_save_csv = "BC_IDC_outputs_a_tsne" + setting_str + ".csv"
        if not os.path.exists(dir_save_csv):
            os.makedirs(dir_save_csv)
        df_outputsA_pca_tsne.to_csv(dir_save_csv + name_save_csv)
        # plot the TSNE result
        colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'grey', 'orange', 'pink']
        for i in range(2):
            plt.scatter(df_outputsA_pca_tsne.loc[i][0], df_outputsA_pca_tsne.loc[i][1], color=colors[i], marker='.')
        plt.title('BC_IDC_ResnetVFL OutputsA TSNE' + setting_str)
        # plt.show()
        name_save_tsne_fig = "BC_IDC_Resnet_VFL_OutputsA_TSNE" + setting_str + ".png"
        plt.savefig(dir_save_csv + name_save_tsne_fig)
        plt.close()

    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()


if __name__ == '__main__':
    main()
