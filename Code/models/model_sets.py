"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init,  BasicBlock
import torch
from models.mixtext import MixText


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet110(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[18, 18, 18], kernel_size=kernel_size, num_classes=num_classes)


def resnet56(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[9, 9, 9], kernel_size=kernel_size, num_classes=num_classes)


class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCinic10(nn.Module):
    def __init__(self):
        super(BottomModelForCinic10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCinic10(nn.Module):
    def __init__(self):
        super(TopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCifar100(nn.Module):
    def __init__(self):
        super(BottomModelForCifar100, self).__init__()
        self.resnet20 = resnet20(num_classes=100)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(200, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(200)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(BottomModelForTinyImageNet, self).__init__()
        self.resnet56 = resnet56(num_classes=200)

    def forward(self, x):
        x = self.resnet56(x)
        return x


class TopModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(TopModelForTinyImageNet, self).__init__()
        self.fc1top = nn.Linear(400, 400)
        self.fc2top = nn.Linear(400, 200)
        self.fc3top = nn.Linear(200, 200)
        self.bn0top = nn.BatchNorm1d(400)
        self.bn1top = nn.BatchNorm1d(400)
        self.bn2top = nn.BatchNorm1d(200)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForImageNet(nn.Module):
    def __init__(self):
        super(BottomModelForImageNet, self).__init__()
        self.resnet56 = resnet56(num_classes=1000)

    def forward(self, x):
        x = self.resnet56(x)
        return x


class TopModelForImageNet(nn.Module):
    def __init__(self):
        super(TopModelForImageNet, self).__init__()
        self.fc1top = nn.Linear(2000, 2000)
        self.fc2top = nn.Linear(2000, 1000)
        self.fc3top = nn.Linear(1000, 1000)
        self.bn0top = nn.BatchNorm1d(2000)
        self.bn1top = nn.BatchNorm1d(2000)
        self.bn2top = nn.BatchNorm1d(1000)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)


class TopModelForYahoo(nn.Module):

    def __init__(self):
        super(TopModelForYahoo, self).__init__()
        self.fc1_top = nn.Linear(20, 10)
        self.fc2_top = nn.Linear(10, 10)
        self.fc3_top = nn.Linear(10, 10)
        self.fc4_top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(10)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        x = self.bn3top(x)
        x = F.relu(x)
        x = self.fc4_top(x)

        return x


class BottomModelForYahoo(nn.Module):

    def __init__(self, n_labels):
        super(BottomModelForYahoo, self).__init__()
        self.mixtext_model = MixText(n_labels, True)

    def forward(self, x):
        x = self.mixtext_model(x)
        return x


D_ = 2 ** 13


class TopModelForCriteo(nn.Module):

    def __init__(self):
        super(TopModelForCriteo, self).__init__()
        self.fc1_top = nn.Linear(8, 8)
        self.fc2_top = nn.Linear(8, 4)
        self.fc3_top = nn.Linear(4, 2)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = F.relu(x)
        x = self.fc1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return x


class BottomModelForCriteo(nn.Module):

    def __init__(self, half=14, is_adversary=False):
        super(BottomModelForCriteo, self).__init__()
        if not is_adversary:
            half = D_ - half
        self.fc1 = nn.Linear(half, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class TopModelForBcw(nn.Module):

    def __init__(self):
        super(TopModelForBcw, self).__init__()
        self.fc1_top = nn.Linear(4, 4)
        self.bn0_top = nn.BatchNorm1d(4)
        self.fc2_top = nn.Linear(4, 2)
        self.bn1_top = nn.BatchNorm1d(4)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0_top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        return x


class BottomModelForBcw(nn.Module):

    def __init__(self, half=14, is_adversary=False):
        super(BottomModelForBcw, self).__init__()
        if not is_adversary:
            half = 28 - half
        self.fc1 = nn.Linear(half, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, half, is_adversary, n_labels=10):
        if self.dataset_name == 'ImageNet':
            return BottomModelForImageNet()
        elif self.dataset_name == 'CIFAR10':
            return BottomModelForCifar10()
        elif self.dataset_name == 'CIFAR100':
            return BottomModelForCifar100()
        elif self.dataset_name == 'TinyImageNet':
            return BottomModelForTinyImageNet()
        elif self.dataset_name == 'CINIC10L':
            return BottomModelForCinic10()
        elif self.dataset_name == 'Yahoo':
            return BottomModelForYahoo(n_labels)
        elif self.dataset_name == 'Criteo':
            return BottomModelForCriteo(half, is_adversary)
        elif self.dataset_name == 'BCW':
            return BottomModelForBcw(half, is_adversary)
        else:
            raise Exception('Unknown dataset name!')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self):
        if self.dataset_name == 'ImageNet':
            return TopModelForImageNet()
        elif self.dataset_name == 'CIFAR10':
            return TopModelForCifar10()
        elif self.dataset_name == 'CIFAR100':
            return TopModelForCifar100()
        elif self.dataset_name == 'TinyImageNet':
            return TopModelForTinyImageNet()
        elif self.dataset_name == 'CINIC10L':
            return TopModelForCinic10()
        elif self.dataset_name == 'Yahoo':
            return TopModelForYahoo()
        elif self.dataset_name == 'Criteo':
            return TopModelForCriteo()
        elif self.dataset_name == 'BCW':
            return TopModelForBcw()
        else:
            raise Exception('Unknown dataset name!')


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)
