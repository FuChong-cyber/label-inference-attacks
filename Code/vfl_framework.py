import argparse
import ast
import os
import time
import dill
from time import time
import sys
sys.path.insert(0, "./")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from datasets import get_dataset
from my_utils import utils
from models import model_sets
import my_optimizers
import possible_defenses

plt.switch_backend('agg')

D_ = 2 ** 13
BATCH_SIZE = 1000


def split_data(data):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    if args.test_upper_bound:
        x_b = torch.zeros_like(x_b)
    return x_a, x_b


class VflFramework(nn.Module):

    def __init__(self):
        super(VflFramework, self).__init__()
        # counter for direct label inference attack
        self.inferred_correct = 0
        self.inferred_wrong = 0
        # bottom model a can collect output_a for label inference attack
        self.collect_outputs_a = False
        self.outputs_a = torch.tensor([]).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.if_collect_training_dataset_labels = False

        # adversarial options
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        # self.defense_ss = args.ss

        # indicates whether to conduct the direct label inference attack
        self.direct_attack_on = False

        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.loss_func_bottom_model = utils.keep_predict_loss

        # bottom model A
        self.malicious_bottom_model_a = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=True
        )
        # bottom model B
        self.benign_bottom_model_b = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=False
        )
        # top model
        self.top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()

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
        if args.dataset != 'Yahoo':
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    self.benign_bottom_model_b.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD(self.benign_bottom_model_b.parameters(),
                                                                 lr=args.lr,
                                                                 momentum=args.momentum,
                                                                 weight_decay=args.weight_decay)
        else:
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    [
                        {"params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    [
                        {"params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    [
                        {"params": self.benign_bottom_model_b.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.benign_bottom_model_b.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD([
                    {"params": self.benign_bottom_model_b.mixtext_model.bert.parameters(), "lr": 5e-6},
                    {"params": self.benign_bottom_model_b.mixtext_model.linear.parameters(), "lr": 5e-4},
                ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

    def forward(self, x):
        # in vertical federated setting, each party has non-lapping features of the same sample
        x_a, x_b = split_data(x)
        out_a = self.malicious_bottom_model_a(x_a)
        out_b = self.benign_bottom_model_b(x_b)
        if args.use_top_model:
            out = self.top_model(out_a, out_b)
        else:
            out = out_a + out_b
        return out

    def simulate_train_round_per_batch(self, data, target):
        timer_mal = 0
        timer_benign = 0
        # simulate: bottom models forward, top model forward, top model backward and update, bottom backward and update

        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        if self.if_collect_training_dataset_labels:
            self.labels_training_dataset = torch.cat((self.labels_training_dataset, target), dim=0)
        # store grad of input of top model/outputs of bottom models
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)

        # --bottom models forward--
        x_a, x_b = split_data(data)

        # make x_b random noise
        # x_b = torch.rand_like(x_b)

        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        start = time()
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        end = time()
        time_cost = end - start
        timer_mal += time_cost
        # bottom model a can collect output_a for label inference attack
        if self.collect_outputs_a:
            self.outputs_a = torch.cat((self.outputs_a, output_tensor_bottom_model_a.data))
        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        start2 = time()
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        end2 = time()
        time_cost2 = end2 - start2
        timer_benign += time_cost2
        # -top model-
        # (we omit interactive layer for it doesn't effect our attack or possible defenses)
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

        if args.use_top_model:
            self.top_model.train(mode=True)
            output_framework = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss_framework = model_sets.update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                                   model=self.top_model,
                                                                   output=output_framework,
                                                                   batch_target=target,
                                                                   loss_func=self.loss_func_top_model)
        else:
            output_framework = input_tensor_top_model_a + input_tensor_top_model_b
            loss_framework = self.loss_func_top_model(output_framework, target)
            loss_framework.backward()

        # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad

        # defenses here: the server(who controls top model) can defend against label inference attack by protecting
        # print("before defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)
        # gradients sent to bottom models
        model_all_layers_grads_list = [grad_output_bottom_model_a, grad_output_bottom_model_b]
        # privacy preserving deep learning
        if self.defense_ppdl:
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_a],
                                         theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_b],
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
        # sign SGD
        # if self.defense_ss:
        #     for tensor in model_all_layers_grads_list:
        #         torch.sign(tensor, out=tensor)
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(model_all_layers_grads_list)
        # print("after defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)

        # server sends back output_tensor_server_a.grad to the adversary (participant a), so
        # the adversary can use this gradient to perform direct label inference attack.
        if self.direct_attack_on:
            for sample_id in range(len(grad_output_bottom_model_a)):
                grad_per_sample = grad_output_bottom_model_a[sample_id]
                for logit_id in range(len(grad_per_sample)):
                    if grad_per_sample[logit_id] < 0:
                        inferred_label = logit_id
                        if inferred_label == target[sample_id]:
                            self.inferred_correct += 1
                        else:
                            self.inferred_wrong += 1
                        break

        # --bottom models backward/update--
        # -bottom model a: backward/update-
        # print("malicious_bottom_model_a")
        start = time()
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_malicious_bottom_model_a,
                                                 model=self.malicious_bottom_model_a,
                                                 output=output_tensor_bottom_model_a,
                                                 batch_target=grad_output_bottom_model_a,
                                                 loss_func=self.loss_func_bottom_model)
        end = time()
        time_cost = end - start
        timer_mal += time_cost
        # -bottom model b: backward/update-
        # print("benign_bottom_model_b")
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                 model=self.benign_bottom_model_b,
                                                 output=output_tensor_bottom_model_b,
                                                 batch_target=grad_output_bottom_model_b,
                                                 loss_func=self.loss_func_bottom_model)
        end2 = time()
        time_cost2 = end2 - end
        timer_benign += time_cost2
        timer_on = False
        if timer_on:
            print("timer_mal:", timer_mal)
            print("timer_benign:", timer_benign)

        return loss_framework


def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def test_per_epoch(test_loader, framework, k=5, loss_func_top_model=None):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.dataset == 'Yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()
            # set all sub-models to eval mode.
            framework.malicious_bottom_model_a.eval()
            framework.benign_bottom_model_b.eval()
            framework.top_model.eval()
            # run forward process of the whole framework
            x_a, x_b = split_data(data)
            output_tensor_bottom_model_a = framework.malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b = framework.benign_bottom_model_b(x_b)

            if args.use_top_model:
                output_framework = framework.top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            else:
                output_framework = output_tensor_bottom_model_a + output_tensor_bottom_model_b

            correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))

            # sum up batch loss
            test_loss += loss_func_top_model(output_framework, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch
            # print("one batch done")
            count += 1
            if int(0.1 * len(test_loader)) > 0:
                count_percent_10 = count // int(0.1 * len(test_loader))
                if count_percent_10 <= 10 and count % int(0.1 * len(test_loader)) == 0 and\
                        count // int(0.1 * len(test_loader)) > 0:
                    print(f'{count // int(0.1 * len(test_loader))}0 % completed...')
                # print(count)

            if args.dataset == 'Criteo' and count == test_loader.train_batches_num:
                break

        if args.dataset == 'Criteo':
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
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)
    if args.dataset == 'Criteo':
        train_loader = train_dataset
        test_loader = test_dataset
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=True,
            # num_workers=args.workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            # num_workers=args.workers
        )
    # check size_bottom_out and num_classes
    if args.use_top_model is False:
        if dataset_setup.size_bottom_out != dataset_setup.num_classes:
            raise Exception('If no top model is used,'
                            ' output tensor of the bottom model must equal to number of classes.')
    return train_loader, test_loader


def main():
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
    if args.test_upper_bound:
        setting_str += "_upperbound"
    setting_str += "_"
    if args.dataset != 'Yahoo':
        setting_str += "half="
        setting_str += str(args.half)
    if not args.use_top_model:
        setting_str += '_NoTopModel'
    print("settings:", setting_str)

    model = VflFramework()
    model = model.cuda()
    cudnn.benchmark = True

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_top_model,
                                                                  milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_m_a = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_malicious_bottom_model_a,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_b_b = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_benign_bottom_model_b,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)

    train_loader, val_loader = set_loaders()

    dir_save_model = args.save_dir + f"/saved_models/{args.dataset}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    # start training. do evaluation every epoch.
    # print('Test the initialized model:')
    # print('Evaluation on the training dataset:')
    # test_per_epoch(test_loader=train_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)
    # print('Evaluation on the testing dataset:')
    # test_per_epoch(test_loader=val_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)
    for epoch in range(args.epochs):
        print('model.optimizer_top_model current lr {:.5e}'.format(model.optimizer_top_model.param_groups[0]['lr']))
        print('model.optimizer_malicious_bottom_model_a current lr {:.5e}'.format(
            model.optimizer_malicious_bottom_model_a.param_groups[0]['lr']))
        print('model.optimizer_benign_bottom_model_b current lr {:.5e}'.format(
            model.optimizer_benign_bottom_model_b.param_groups[0]['lr']))

        if epoch == 0:
            model.direct_attack_on = True
        else:
            model.direct_attack_on = False

        if epoch == args.epochs - 1 and args.if_cluster_outputsA:
            model.collect_outputs_a = True
            model.if_collect_training_dataset_labels = True
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.dataset == 'Yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()
            loss_framework = model.simulate_train_round_per_batch(data, target)
            if batch_idx % 25 == 0:
                if args.dataset == 'Criteo':
                    num_samples = len(train_loader) * BATCH_SIZE
                else:
                    num_samples = len(train_loader.dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_samples,
                           100. * batch_idx / len(train_loader), loss_framework.data.item()))
        lr_scheduler_top_model.step()
        lr_scheduler_m_a.step()
        lr_scheduler_b_b.step()

        if epoch == args.epochs - 1:
            txt_name = f"{args.dataset}_saved_framework{setting_str}"
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Evaluation on the training dataset:')
                test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)
                print('Evaluation on the testing dataset:')
                test_per_epoch(test_loader=val_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)

                if not args.use_top_model:
                    # performance of the direct label inference attack
                    print("inferred correctly:", model.inferred_correct)
                    if args.dataset == 'Criteo':
                        num_samples = len(train_loader) * BATCH_SIZE
                    else:
                        num_samples = len(train_loader.dataset)
                    num_all_train_samples = num_samples
                    print("all:", num_all_train_samples)
                    print("Direct label inference accuracy:", model.inferred_correct / num_all_train_samples)
                    print("Direct label inference attack evaluated...")

                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

        print('Evaluation on the training dataset:')
        test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                       loss_func_top_model=model.loss_func_top_model)
        print('Evaluation on the testing dataset:')
        test_per_epoch(test_loader=val_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)

    # save model
    torch.save(model, os.path.join(dir_save_model, f"{args.dataset}_saved_framework{setting_str}.pth"),
               pickle_module=dill)

    if args.if_cluster_outputsA:
        outputsA_list = model.outputs_a.detach().clone().cpu().numpy().tolist()
        labels_list = model.labels_training_dataset.detach().clone().cpu().numpy().tolist()
        # plot TSNE cluster result
        outputsA_pca_tsne = TSNE()
        outputsA_pca_tsne.fit_transform(outputsA_list)
        df_outputsA_pca_tsne = pd.DataFrame(outputsA_pca_tsne.embedding_, index=labels_list)
        # plot the TSNE result
        colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'grey', 'orange', 'pink']
        # get num_classes
        dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
        num_classes = dataset_setup.num_classes
        for i in range(num_classes):
            plt.scatter(df_outputsA_pca_tsne.loc[i][0], df_outputsA_pca_tsne.loc[i][1], color=colors[i], marker='.')
        plt.title('VFL OutputsA TSNE' + setting_str)
        # plt.show()
        dir_save_tsne_pic = args.save_dir + f"/csv_files/{args.dataset}_csv_files"
        if not os.path.exists(dir_save_tsne_pic):
            os.makedirs(dir_save_tsne_pic)
        df_outputsA_pca_tsne.to_csv(
            dir_save_tsne_pic + f"/{args.dataset}_outputs_a_tsne{setting_str}.csv")
        plt.savefig(os.path.join(dir_save_tsne_pic, f"{args.dataset}_Resnet_VFL_OutputsA_TSNE{setting_str}.png"))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('-d', '--dataset', default='Criteo', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='D:/Datasets/yahoo_answers_csv/')
    '''
    'D:/Datasets/CIFAR10'
    'D:/Datasets/CIFAR100'
    'D:/Datasets/TinyImageNet'
    'D:/Datasets/CINIC10L'
    'D:/Datasets/BC_IDC'
    'D:/Datasets/Criteo/criteo1e?.csv'
    'D:/Datasets/yahoo_answers_csv/'
    'D:/Datasets/BreastCancerWisconsin/wisconsin.csv'
    '''
    # framework paras
    parser.add_argument('--use-top-model', help='vfl framework has top model or not. If no top model'
                                                'is used, automatically turn on direct label inference attack,'
                                                'and report label inference accuracy on the training dataset',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--test-upper-bound', help='if set to True, test the upper bound of our attack: if all the'
                                                   'adversary\'s samples are labeled, how accurate is the adversary\'s '
                                                   'label inference ability?',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA',
                        type=ast.literal_eval, default=True)
    # attack paras
    parser.add_argument('--use-mal-optim',
                        help='whether the attacker uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-all',
                        help='whether all participants use the malicious optimizer. If set to '
                             'True, use_mal_optim will be automatically set to True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-top',
                        help='whether the server(top model) uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    # saving path paras
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='./saved_experiment_results', type=str)
    # possible defenses on/off paras
    parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gc', help='turn_on_gradient_compression',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
    # paras about possible defenses
    parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                        type=float, default=0.75)
    parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.75)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=1e-3)
    parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
    parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)
    # training paras
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of datasets loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')  # TinyImageNet=5e-2, Yahoo=1e-3
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
    if args.use_mal_optim_all:
        args.use_mal_optim = True
    main()
