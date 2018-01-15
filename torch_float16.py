import numpy as np
import argparse
from datetime import datetime
import torch


def build_img(args):
    img = np.random.rand(args.batch_size, args.nin)
    if args.mixf:
        img = torch.Tensor(img).cuda().half()
    else:
        img = torch.Tensor(img).cuda()
    return img


class BuildModel(object):
    def __init__(self, args):
        self.sizes = ([args.nin] + ([args.layer_neurons] * args.layers) +
                      [args.nout])
        self.weights = []
        for i in range(1, len(self.sizes)):
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


def run_benchmark(args):
    img = build_img(args)
    model = BuildModel(args)
    # Start profiling
    time_start = datetime.now()
    for i in range(args.nsteps):
        model.run(img)
        if (i + 1) % 100 == 0:
            print("Step %d/%d " % (i + 1, args.nsteps))
    time_end = datetime.now()  # end profiling
    time_spent = time_end - time_start
    seconds = time_spent.seconds + time_spent.days * 24 * 3600
    profile_message = 'execution time: %s sec + %s microsec' % (seconds, time_spent.microseconds)
    print (profile_message)


if __name__ == '__main__':
    np.random.seed(12345678)
    default_batch_size = 10000
    default_nin = 2000
    default_nout = 2000
    default_nsteps = 1000
    default_layers = 5
    default_layer_neurons = 4000
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

    args = parser.parse_args()
    run_benchmark(args)
