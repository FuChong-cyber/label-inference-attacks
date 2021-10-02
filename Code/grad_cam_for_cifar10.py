import torch
from torch.autograd import Variable
from torch.autograd import Function
import cv2
from torchvision import datasets, transforms
import numpy as np
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dill


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradientsA = []
        self.gradientsB = []

    def save_gradientA(self, grad):
        self.gradientsA.append(grad)

    def save_gradientB(self, grad):
        self.gradientsB.append(grad)

    def __call__(self, x):
        outputsA = []
        outputsB = []
        self.gradientsA = []
        self.gradientsB = []
        xA = x[:, :, :, 0:16]
        xB = x[:, :, :, 16:32]

        xA = self.model.malicious_bottom_model_a.resnet20.conv1(xA)
        xA.register_hook(self.save_gradientA)
        outputsA += [xA]
        xA = F.relu(self.model.malicious_bottom_model_a.resnet20.bn1(xA))
        xA = self.model.malicious_bottom_model_a.resnet20.layer1(xA)
        xA = self.model.malicious_bottom_model_a.resnet20.layer2(xA)
        xA = self.model.malicious_bottom_model_a.resnet20.layer3(xA)
        xA = F.avg_pool2d(xA, xA.size()[2:])
        xA = xA.view(xA.size(0), -1)
        xA = self.model.malicious_bottom_model_a.resnet20.linear(xA)

        xB = self.model.benign_bottom_model_b.resnet20.conv1(xB)
        xB.register_hook(self.save_gradientB)
        outputsB += [xB]
        xB = F.relu(self.model.benign_bottom_model_b.resnet20.bn1(xB))
        xB = self.model.benign_bottom_model_b.resnet20.layer1(xB)
        xB = self.model.benign_bottom_model_b.resnet20.layer2(xB)
        xB = self.model.benign_bottom_model_b.resnet20.layer3(xB)
        xB = F.avg_pool2d(xB, xB.size()[2:])
        xB = xB.view(xB.size(0), -1)
        xB = self.model.benign_bottom_model_b.resnet20.linear(xB)

        out = self.model.top_model(xA, xB)

        return outputsA, outputsB, out


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradientsA(self):
        return self.feature_extractor.gradientsA

    def get_gradientsB(self):
        return self.feature_extractor.gradientsB

    def __call__(self, x):
        target_activationsA, target_activationsB, output = self.feature_extractor(x)
        return target_activationsA, target_activationsB, output


def preprocess_image(img):
    preprocessed_img = img.copy()
    preprocessed_img = \
        np.ascontiguousarray(preprocessed_img)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, is_normal):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img)
    img = np.transpose(img, (1, 2, 0))
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam_show = cv2.resize(cam, (364, 364), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("cam.jpg", cam_show)
    cv2.waitKey(0)
    if is_normal:
        cv2.imwrite("./GradCAM/cam_normal.jpg", cam_show)
    else:
        cv2.imwrite("./GradCAM/cam_mal.jpg", cam_show)


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            featuresA, featuresB, output = self.extractor(input.cuda())
        else:
            featuresA, featuresB, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_valA = self.extractor.get_gradientsA()[-1].cpu().data.numpy()
        grads_valB = self.extractor.get_gradientsB()[-1].cpu().data.numpy()

        targetA = featuresA[-1]
        targetA = targetA.cpu().data.numpy()[0, :]
        targetB = featuresB[-1]
        targetB = targetB.cpu().data.numpy()[0, :]

        weightsA = np.mean(grads_valA, axis=(2, 3))[0, :]
        camA = np.zeros(targetA.shape[1:], dtype=np.float32)
        weightsB = np.mean(grads_valB, axis=(2, 3))[0, :]
        camB = np.zeros(targetB.shape[1:], dtype=np.float32)

        for i, w in enumerate(weightsA):
            camA += w * targetA[i, :, :]
        for i, w in enumerate(weightsB):
            camB += w * targetB[i, :, :]

        print("camA.shape:", camA.shape)
        print("camB.shape:", camB.shape)
        cam = np.concatenate((camA, camB), axis=1)
        print("cam.shape:", cam.shape)

        cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./test_image.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def get_img_cifar10(img_number):
    # cifar10 digits dataset
    train_dataset = datasets.CIFAR10(root='D:/Datasets/CIFAR10',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)

    def imshow(img):
        import matplotlib
        matplotlib.use('TkAgg')
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(npimg)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig("./coriginal.jpg", bbox_inches='tight', pad_inches=0)
        plt.show()

    dataiter = iter(train_loader)
    id = 0
    while (1):
        images, labels = dataiter.next()
        if id == img_number:
            print(images.shape)  # [bs,1,32,32]

            # random_b
            # images[:, :, :, 16:32] = torch.rand_like(images[:, :, :, 16:32])

            imshow(images[0])
            # imshow(torchvision.utils.make_grid(images))
            return images.numpy()[0]
        id += 1


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    sample_id_in_cifar10 = 162
    file_name = "cifar10_saved_framework_lr=0.1_normal.pth"
    model = torch.load("D:/MyCodes/label_inference_attacks_against_vfl/saved_experiment_results/"
                       "saved_models/CIFAR10_saved_models/" + file_name, pickle_module=dill)
    print(model)
    grad_cam = GradCam(model, target_layer_names=["layer4"], use_cuda=args.use_cuda)
    img = get_img_cifar10(sample_id_in_cifar10)
    input = preprocess_image(img)
    print('input.size()=', input.size())
    target_index = None

    mask = grad_cam(input, target_index)

    print("img shape:", img.shape)
    print("mask shape:", mask.shape)
    is_normal = ("normal" in file_name)
    show_cam_on_image(img, mask, is_normal)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    # utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
# utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
