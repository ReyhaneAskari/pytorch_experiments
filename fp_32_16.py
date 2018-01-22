from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--mixf', action='store_true', default=False,
                    help='enables using mixed float precision')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        # input, output, kernel
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=5)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=5)
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 100)

    def forward(self, x):
        # (64, 3, 128, 128)
        x = F.relu(self.conv1(x))
        # [64, 128, 63, 63]
        x = F.relu(self.conv2(x))
        # 64, 512, 14, 14
        x = F.relu(self.conv3(x))
        # 64 , 512, 2, 2
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()
if args.mixf:
    model.cuda().half()
else:
    model.cuda()

if args.mixf:
    params_copy = [param.clone().
                   type(torch.cuda.FloatTensor).detach() for param
                   in model.parameters()]
    for param in params_copy:
        param.requires_grad = True

    optimizer = optim.SGD(params_copy, lr=args.lr, momentum=0.9)
else:
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9)


def set_grad(params, params_with_grad):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.
                                            data.new().
                                            resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


def train(epoch):
    model.train()
    # dummy dataset the same size as imagenet
    data_ = torch.FloatTensor(np.random.randn(64, 3, 128, 128))
    target_ = torch.FloatTensor(np.random.randint(0, 128, (64)))
    total_forward = 0
    for batch_idx in range(1000):
        if args.mixf:
            data, target = data_.cuda().half(), target_.cuda().half()
        else:
            data, target = data_.cuda(), target_.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        t_0 = time.time()
        output = model(data)
        total_forward += time.time() - t_0
        # dummy loss
        loss = 0.0 * (torch.sum(output) + torch.sum(target))
        loss.backward()

        if args.mixf:
            set_grad(params_copy, list(model.parameters()))
            optimizer.step()
            params = list(model.parameters())
            for i in range(len(params)):
                params[i].data.copy_(params_copy[i].data)
        else:
            optimizer.step()

        if batch_idx % 100 == 0:
            print('\tbatch_idx: ' + str(batch_idx))
    print(total_forward)

for epoch in range(1, args.epochs + 1):
    train(epoch)
