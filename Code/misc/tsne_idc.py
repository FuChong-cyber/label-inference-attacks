from __future__ import print_function

import argparse
import os
os.sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.nn.parallel

import dill
import copy
from 即将删除的文件 import BC_IDC_loader2

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
parser.add_argument('--path-dataset', help='path_dataset',
                    type=str, default='E:/Dataset/BC_IDC_muscle')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--party-num', help='party-num',
                    type=int, default=2)
parser.add_argument('--resume',
                    default='./saved_models/BC_IDC_saved_models/BC_IDC_saved_framework22-layers_lr=0.1_mal_party-num=2.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', )
# './saved_models/BC_IDC_saved_models/BC_IDC_saved_framework22-layers_lr=0.1_mal_party-num=2.pth'
# './saved_models/BC_IDC_saved_models/BC_IDC_saved_framework_lr=0.1_normal_party-num=2.pth'

args = parser.parse_args()

train_loader = BC_IDC_loader2.IdcDataset(args.path_dataset, args.batch_size, args.party_num).get_trainloader()

# Resume
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
args.out = os.path.dirname(args.resume)
checkpoint = torch.load(args.resume, pickle_module=dill)
# print("checkpoint:", checkpoint.malicious_bottom_model_a)
model = copy.deepcopy(checkpoint.bottom_models[0])
print(model)

outputs_a = torch.tensor([]).cuda()
labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()

train_set_num = len(train_loader)
current_num = 0

for data, target in train_loader:
    data = data.cuda()
    target = target.cuda()
    data = data[:, 0:1, :, :, :].squeeze(1)
    with torch.no_grad():
        output = model(data)
    # print(target)
    outputs_a = torch.cat((outputs_a, output))
    labels_training_dataset = torch.cat((labels_training_dataset, target), dim=0)

    current_num += 1
    if current_num % 100 == 0:
        print("Progress:", current_num, "/", train_set_num, "->", current_num/train_set_num)

    if current_num % 10000 == 0:
        break

outputs_a = outputs_a.cpu().tolist()
labels_training_dataset = labels_training_dataset.cpu().tolist()

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
labeled_train_data_a_tsne = TSNE()
labeled_train_data_a_tsne.fit_transform(outputs_a)
df_train_data_a_tsne = pd.DataFrame(labeled_train_data_a_tsne.embedding_, index=labels_training_dataset)
# plot the TSNE result
#           0        1        2        3        4       5       6           7             8        9
colors = ['black', 'red', 'yellow', 'green', 'cyan', 'silver', 'purple', 'saddlebrown', 'orange', 'pink']
for i in range(2):
    plt.scatter(df_train_data_a_tsne.loc[i][0], df_train_data_a_tsne.loc[i][1], color=colors[i], marker='.')
plt.title('labeled_train_data_a_tsne')
plt.show()