from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

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
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=5)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=5)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=5)
        self.fc1 = nn.Linear(4608, 2000)
        self.fc2 = nn.Linear(2000, 100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2(x), 4))
        x = F.relu(F.max_pool2d(self.conv3(x), 4))
        # import ipdb; ipdb.set_trace()
        x = x.view(-1, 4608)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


def train(epoch):
    model.train()
    # dummy dataset the same size as imagenet
    data_ = torch.FloatTensor(np.random.randn(64, 3, 224, 224))
    target_ = torch.LongTensor(np.random.randint(0, 100, (64)))
    for batch_idx in range(1000):
        data, target = data_.cuda(), target_.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('\tbatch_idx: ' + str(batch_idx))

for epoch in range(1, args.epochs + 1):
    train(epoch)
