import sys
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import sys

import cv2
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os

import config as conf
import tensorflow as tf
from Architectures.impl.HardNet import HardNet
from Networks.impl.standard import StandardNetwork


# all types of patches
tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5', \
       't1', 't2', 't3', 't4', 't5']


class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps

    def __init__(self, base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t + '.png')
            im = cv2.imread(im_path, 0)
            self.N = im.shape[0] / 65
            setattr(self, t, np.split(im, self.N))


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x

try:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    seqs = glob.glob(sys.argv[1] + '/*')
    seqs = [os.path.abspath(p) for p in seqs]
except:
    print(
        'Wrong input format. Try python hpatches_extract_HardNet.py /home/ubuntu/dev/hpatches/hpatches-benchmark/data/hpatches-release /home/old-ufo/dev/hpatches/hpatches-benchmark/data/descriptors')
    sys.exit(1)

w = 65


def HardNet_forward(model, input):
    """ Includes input and output norm stuff """
    input_norm = HardNet.start(input=input)
    x_features = model(input=input_norm)
    return HardNet.end(x_features)


# Change accordingly
checkpoint_path = conf.ckpt_dir + "HardNet/checkpoint_0.ckpt"
# Where the descriptor results are saved
output_dir = "..."
# Where hpatches-release is located
input_dir = "..."

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create the architecture
architecture = HardNet()
# Use standard network
model = StandardNetwork(architecture=architecture)
model.build("StandardNetwork_HardNet")

# Input placeholder
input_patches = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])

# Output node
out_op = HardNet_forward(model, input=input_patches)

with tf.Session() as sess:

    # Load from checkpoint
    saver.restore(sess, checkpoint_path)

    for seq_path in seqs:
        seq = hpatches_sequence(seq_path)

        descr = np.zeros((seq.N, 128))  # trivial (mi,sigma) descriptor
        for tp in tps:
            print(seq.name + '/' + tp)

            if os.path.isfile(os.path.join(output_dir, tp + '.csv')):
                continue

            n_patches = 0

            for i, patch in enumerate(getattr(seq, tp)):
                n_patches += 1

            t = time.time()
            patches_for_net = np.zeros((n_patches, 1, 32, 32))

            for i, patch in enumerate(getattr(seq, tp)):
                patches_for_net[i, 0, :, :] = cv2.resize(patch[0:w, 0:w], (32, 32))

            outs = []
            bs = 128
            n_batches = n_patches / bs + 1
            for batch_idx in range(n_batches):
                st = batch_idx * bs

                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs

                if st >= end:
                    continue

                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)

                # Compute output
                out_a = sess.run(out_op, feed_dict={input_patches: data_a})

                outs.append(out_a.data.cpu().numpy().reshape(-1, 128))

            res_desc = np.concatenate(outs)
            print(res_desc.shape, n_patches)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))

            np.savetxt(os.path.join(output_dir, tp + '.csv'), out, delimiter=',', fmt='%10.5f')  # X is an array
