import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable


def build_img(args):
    if args.conv:
        img = np.random.rand(args.batch_size, args.input_channel,
                             args.image_width, args.image_width)
    else:
        img = np.random.rand(args.batch_size, args.nin)
    if args.mixf:
        img = torch.Tensor(img).cuda().half()
    else:
        img = torch.Tensor(img).cuda()
    return Variable(img)


class BuildModel(object):
    def __init__(self, args):
        if args.conv:
            self.sizes = ([args.input_channel] +
                          [args.hidden_channel] * (args.layers + 1))
            self.convs = []
        else:
            self.sizes = ([args.nin] + ([args.layer_neurons] * args.layers) +
                          [args.nout])
            self.weights = []
        for i in range(1, len(self.sizes)):
            if args.conv:
                if args.mixf:
                    self.convs += [nn.Conv2d(self.sizes[i - 1],
                                             self.sizes[i],
                                             kernel_size=args.kernel_size).cuda().half()]
                else:
                    self.convs += [nn.Conv2d(self.sizes[i - 1],
                                             self.sizes[i],
                                             kernel_size=args.kernel_size).cuda()]
            else:
                shape = (self.sizes[i - 1], self.sizes[i])
                init_value = np.random.rand(*shape)
                if args.mixf:
                    self.weights += [torch.Tensor(init_value).cuda().half()]
                else:
                    self.weights += [torch.Tensor(init_value).cuda()]

    def run(self, img):
        self.outputs = [img]
        for i in range(1, len(self.sizes)):
            out = torch.mm(self.outputs[-1], self.weights[i - 1])
            self.outputs.append(out)
        return self.outputs[-1]

    def run_conv(self, img):
        self.outputs = [img]
        for i in range(1, len(self.sizes)):
            out = self.convs[i - 1](self.outputs[-1])
            self.outputs.append(out)
        return self.outputs[-1]


def run_benchmark(args):
    img = build_img(args)
    model = BuildModel(args)
    # Start profiling
    time_start = datetime.now()
    if args.conv:
        run_fnc = model.run_conv
    else:
        run_fnc = model.run
    for i in range(args.nsteps):
        run_fnc(img)
        if (i + 1) % 100 == 0:
            print("Step %d/%d " % (i + 1, args.nsteps))
    time_end = datetime.now()  # end profiling
    time_spent = time_end - time_start
    seconds = time_spent.seconds + time_spent.days * 24 * 3600
    profile_message = 'execution time: %s sec + %s microsec' % (seconds, time_spent.microseconds)
    print (profile_message)


if __name__ == '__main__':
    np.random.seed(12345678)
    default_batch_size = 4096
    default_nin = 2048
    default_nout = 2048
    default_nsteps = 2000
    default_layers = 1
    default_layer_neurons = 2048

    default_input_channel = 3
    default_hidden_channel = 128
    default_kernel_size = 5
    default_image_size = 64

    parser = argparse.ArgumentParser()
    parser.add_argument('--mixf', action='store_true', default=False,
                        help='Enables mixed float precision')
    parser.add_argument("--batch_size", type=int, default=default_batch_size,
                        help='Batch size of the layer (default %d)' % default_batch_size)
    parser.add_argument("--nin", type=int, default=default_nin,
                        help='Input size of the layer (default %d)' % default_nin)
    parser.add_argument("--nout", type=int, default=default_nout,
                        help='Output size of the layer (default %d)' % default_nout)
    parser.add_argument("--nsteps", type=int, default=default_nsteps,
                        help='Number of training steps (default %d)' % default_nsteps)
    parser.add_argument("--layers", type=int, default=default_layers,
                        help='Number of layers (default %d)' % default_layers)
    parser.add_argument("--layer-neurons", type=int, default=default_layer_neurons,
                        help='Number of neurons per layer (default %d)' % default_layer_neurons)
    parser.add_argument("--conv", action='store_true', default=False)
    parser.add_argument("--input_channel", type=int, default=default_input_channel)
    parser.add_argument("--hidden_channel", type=int, default=default_hidden_channel)
    parser.add_argument("--kernel_size", type=int, default=default_kernel_size)
    parser.add_argument("--image_width", type=int, default=default_image_size)

    args = parser.parse_args()
    run_benchmark(args)
