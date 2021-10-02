from torch.optim.optimizer import Optimizer
import torch
import matplotlib.pyplot as plt
from time import time

last_whole_model_params_list = []
new_whole_model_params_list = []
batch_cos_list = []
near_minimum = False


class MaliciousSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0

        self.certain_grad_ratios = torch.tensor([])

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        # print("malicious optim stepping")

        loss = None
        if closure is not None:
            loss = closure()

        # show grad cos
        # global last_whole_model_params_list, new_whole_model_params_list, batch_cos_list, near_minimum
        # new_whole_model_params_list = []

        id_group = 0
        for i in range(len(self.param_groups)):
            self.last_parameters_grads.append([])

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            start = time()
            # Indicate which module's paras we are processing
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                # if id_parameter == 0:
                #     print("before malicious ops")
                #     print("params:", p[:2])
                #     print("params.grad:", p.grad[:2])
                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p.grad.data)
                    if nesterov:
                        p.grad.data = p.grad.data.add(momentum, buf)
                    else:
                        p.grad.data = buf

                # show grad cos
                # new_whole_model_params_list.extend(p.grad.clone().detach().cpu().numpy().flatten())
                if not near_minimum:
                    # print("MalSGD")
                    if len(self.last_parameters_grads[id_group]) <= id_parameter:
                        # print("first batch")
                        # print("id_params:", id_parameter)
                        # print("p.grad.shape", p.grad.shape)
                        self.last_parameters_grads[id_group].append(p.grad.clone().detach())
                    else:
                        # print("2nd-> epoch")
                        last_parameter_grad = self.last_parameters_grads[id_group][id_parameter]

                        # 至此我们得到了model A中一个模块的上次(滑动平均后的)梯度矩阵和本次(滑动平均后的)梯度矩阵，且两者都是tensor格式
                        current_parameter_grad = p.grad.clone().detach()
                        ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (current_parameter_grad / (last_parameter_grad + 1e-7))
                        # print("ratio_grad_scale_up:", ratio_grad_scale_up)
                        # 小梯度扩增倍数修正:
                        # 太小的参数更新会导致本次梯度扩增倍数太大，因此ratio_temp矩阵中，根据较小的last_para计算来的倍数要作废
                        # 重新设置为1.0
                        # background_tensor = torch.ones(ratio_grad_scale_up.shape).to(torch.float)
                        # if 'cuda' in str(ratio_grad_scale_up.device):
                        #     background_tensor = background_tensor.cuda()
                        # ratio_grad_scale_up = torch.where(abs(last_parameter_grad) > self.min_grad_to_process,
                        #                                   ratio_grad_scale_up,
                        #                                   background_tensor)
                        # 把动态倍率中小于min的部分改成min，大于max的部分改成max
                        ratio_grad_scale_up = torch.clamp(ratio_grad_scale_up, self.min_ratio, self.max_ratio)
                        # stop = time()
                        # print("tensor computation time:", str(stop - start))
                        # print("pruned ratio_temp:", ratio_grad_scale_up)

                        # if id_parameter == 0:
                        #     self.certain_grad_ratios.append(ratio_grad_scale_up.flatten()[0])
                        #     if len(self.certain_grad_ratios) > 100:
                        #         plt.plot(list(range(len(self.certain_grad_ratios[0:100]))),
                        #                  self.certain_grad_ratios[0:100], c="blue")
                        #         plt.title("model.certain_grad_ratios")
                        #         plt.ylim(ymax=5)
                        #         plt.show()
                        #         exit()
                        # print("current_parameter.grad\n", p.grad[0])
                        p.grad.mul_(ratio_grad_scale_up)
                        # print("current_parameter.grad mul\n", p.grad[0])
                        # exit()
                end = time()
                # print("mal sgd cost:", str(end - start))

                # if id_parameter == 0:
                #     print("after malicious ops")
                #     print("params:", p[:2])
                #     print("params.grad:", p.grad[:2])

                current_parameter_grad = p.grad.clone().detach()
                self.last_parameters_grads[id_group][id_parameter] = current_parameter_grad

                p.data.add_(-group['lr'], p.grad.data)

                id_parameter += 1
            id_group += 1

        return loss


class MaliciousSignSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process

        self.certain_grad_ratios = []

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Indicate which module's paras we are processing
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                # if id_parameter == 0:
                #     print("before malicious ops")
                #     print("params:", p[:2])
                #     print("params.grad:", p.grad[:2])
                if len(self.last_parameters_grads) <= id_parameter:
                    # print("first batch")
                    # print("id_params:", id_parameter)
                    self.last_parameters_grads.append(p.grad.clone().detach().numpy())
                else:
                    last_parameter_grad = self.last_parameters_grads[id_parameter]
                    current_parameter_grad = p.grad.clone().detach().numpy()
                    # 至此我们得到了model A中一个模块的上次梯度矩阵和本次梯度矩阵，且两者都是np格式
                    ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (current_parameter_grad / (last_parameter_grad + 1e-7))
                    grad_shape = current_parameter_grad.shape
                    last_parameter_grad = last_parameter_grad.flatten()
                    grad_length = len(last_parameter_grad)
                    ratio_grad_scale_up = ratio_grad_scale_up.flatten()
                    for i in range(grad_length):
                        # 小梯度扩增倍数修正:
                        # 太小的参数更新会导致本次梯度扩增倍数太大，因此ratio_temp矩阵中，根据较小的last_para计算来的倍数要作废
                        # 重新设置为1.0
                        if abs(last_parameter_grad[i]) < self.min_grad_to_process:
                            ratio_grad_scale_up[i] = 1.0
                        # 把动态倍率中小于min的部分改成min
                        ratio_grad_scale_up[i] = max(ratio_grad_scale_up[i], 1.0)
                        ratio_grad_scale_up[i] = min(ratio_grad_scale_up[i], 5.0)
                    ratio_grad_scale_up = ratio_grad_scale_up.reshape(grad_shape)
                    # print("pruned ratio_temp:", ratio_grad_scale_up)

                    self.last_parameters_grads[id_parameter] = current_parameter_grad

                    # if id_parameter == 0:
                    #     self.certain_grad_ratios.append(ratio_grad_scale_up.flatten()[0])
                    #     if len(self.certain_grad_ratios) > 100:
                    #         plt.plot(list(range(len(self.certain_grad_ratios[0:100]))),
                    #                  self.certain_grad_ratios[0:100], c="blue")
                    #         plt.title("model.certain_grad_ratios")
                    #         plt.ylim(ymax=5)
                    #         plt.show()
                    #         exit()
                    # print("current_parameter.grad\n", p.grad[0])
                    p.grad = p.grad.mul(torch.tensor(ratio_grad_scale_up))
                    # print("current_parameter.grad mul\n", p.grad[0])
                    # exit()

                # if id_parameter == 0:
                #     print("after malicious ops")
                #     print("params:", p[:2])
                #     print("params.grad:", p.grad[:2])

                id_parameter += 1

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # sign SGD only uses sign of gradient to update model
                torch.sign(d_p, out=d_p)
                p.data.add_(-group['lr'], d_p)

        return loss


class SignSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # sign SGD only uses sign of gradient to update model
                torch.sign(d_p, out=d_p)
                p.data.add_(-group['lr'], d_p)

        return loss