'''
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init,  BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, 5)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(kernel_size=(3, 3)):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size)


class BottomModel(nn.Module):
    def __init__(self):
        super(BottomModel, self).__init__()
        self.resnet20 = resnet20()

    def forward(self, x):
        x = self.resnet20(x)
        return x


class BottomModelForDirect(nn.Module):
    def __init__(self):
        super(BottomModelForDirect, self).__init__()
        self.resnet20 = resnet20()
        self.linear = nn.Linear(5, 2)

    def forward(self, x):
        x = self.resnet20(x)
        x = self.linear(x)
        return x


class TopModel(nn.Module):
    def __init__(self, dims_in):
        super(TopModel, self).__init__()
        self.fc1top = nn.Linear(dims_in, 10)
        # self.fc1top = nn.Linear(55, 10)
        self.fc2top = nn.Linear(10, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 2)
        self.bn0top = nn.BatchNorm1d(dims_in)
        # self.bn0top = nn.BatchNorm1d(55)
        self.bn1top = nn.BatchNorm1d(10)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return x


class ResNetOverlap(nn.Module):
    def __init__(self, block, num_blocks, kernel_size):
        super(ResNetOverlap, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, 5)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BottomModelOverlap(nn.Module):
    def __init__(self):
        super(BottomModelOverlap, self).__init__()
        self.resnet = ResNetOverlap(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=(3, 3))

    def forward(self, x):
        x = self.resnet(x)
        return x


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    # print("top loss:", loss)
    # print('output: ', output)
    # print('target: ', target)
    optimizer.zero_grad()  # 所有参数的梯度清零
    loss.backward()  # 即反向传播求梯度
    # for name, params in model.named_parameters():
    #     print("names:", name)
    #     print("params:", params[:5])
    #     print("params.grad:", params.grad[:5])
    optimizer.step()  # 调用optimizer进行梯度下降更新参数
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    # print("bottom loss:", loss)
    # print('output: ', output)
    # print('target: ', target)
    optimizer.zero_grad()  # 所有参数的梯度清零
    loss.backward()  # 即反向传播求梯度
    # for name, params in model.named_parameters():
    #     if "bn1" in name:
    #         print("names:", name)
    #         print("params:", params[:5])
    #         print("params.grad:", params.grad[:5])

    optimizer.step()  # 调用optimizer进行梯度下降更新参数
    return


# def test(net):
#     import numpy as np
#     total_params = 0
#
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.datasets.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.datasets.size())>1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()
