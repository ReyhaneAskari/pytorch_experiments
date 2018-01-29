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
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
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
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

model = Net()

if args.mixf:
    model.cuda().half()
else:
    model.cuda()

ITERS = 300

def train(epoch):
    model.train()
    # dummy dataset the same size as imagenet
    data_ = torch.FloatTensor(np.random.randn(4096, 2048, 1, 1))

    #lets get copy time out of conv time:
    if args.mixf:
        data = data_.cuda().half()
    else:
        data = data_.cuda()

    #time the entire thing, with proper cuda synchronization
    torch.cuda.synchronize()
    start = time.time()

    for batch_idx in range(ITERS):
        output = model(Variable(data))

    torch.cuda.synchronize()
    print("Time / iteration: ", (time.time()-start)/ITERS)

for epoch in range(1, args.epochs + 1):
    train(epoch)
