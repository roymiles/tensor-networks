#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import copy
import PIL
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from W1BS import w1bs_extract_descs_and_save
from Utils import cv2_scale, np_reshape, str2bool
import tensorflow as tf
from Architectures.impl.HardNet import HardNet, conv_ranks, fc_ranks
from Networks.impl.standard import StandardNetwork
from Networks.impl.sandbox import TuckerNet
import config as conf
from base import tfvar_size

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def CorrelationPenaltyLoss(input):
    exit("Add to the laterbase")
    mean1 = tf.math.reduce_mean(input, dim=0)
    zeroed = input - mean1.expand_as(input)
    cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
    d = torch.diag(torch.diag(cor_mat))
    no_diag = cor_mat - d
    d_sq = no_diag * no_diag
    return torch.sqrt(d_sq.sum()) / input.size(0)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--w1bsroot', type=str,
                    default='data/sets/wxbs-descriptors-benchmark/code/',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default='liberty_train/',
                    help='experiment path')
parser.add_argument('--training-set', default='liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--loss', default='triplet_margin',   # triplet-margin is default
                    help='Other options: triplet_margin, softmax, contrastive')
parser.add_argument('--batch-reduce', default='min',
                    help='Other options: min, average, random, random_global, L2Net')
parser.add_argument('--num-workers', default=0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--decor', type=str2bool, default=False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=str2bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor', type=str2bool, default=False,
                    help='use gor')
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--fliprot', type=str2bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--augmentation', type=str2bool, default=False,
                    help='turns on shift and small scale rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Device options
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

suffix = '{}_{}_{}'.format(args.experiment_name, args.training_set, args.batch_reduce)

if args.gor:
    suffix = suffix + '_gor_alpha{:1.1f}'.format(args.alpha)
if args.anchorswap:
    suffix = suffix + '_as'
if args.anchorave:
    suffix = suffix + '_av'
if args.fliprot:
    suffix = suffix + '_fliprot'

triplet_flag = (args.batch_reduce == 'random_global') or args.gor

dataset_names = ['liberty', 'notredame', 'yosemite']

TEST_ON_W1BS = False
# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs

    TEST_ON_W1BS = True


# set random seeds
random.seed(args.seed)
tf.random.set_random_seed(args.seed)
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

# Instead of placeholder, we manually increment
global_step = 0


class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None,load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0, 2, 1)
                img_p = img_p.permute(0, 2, 1)
                if self.out_triplets:
                    img_n = img_n.permute(0, 2, 1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:, :, ::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:, :, ::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:, :, ::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


def create_loaders(load_random_triplets=False):
    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_test = transforms.Compose([
        transforms.Lambda(np_reshape64),
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor()])
    transform_train = transforms.Compose([
        transforms.Lambda(np_reshape64),
        transforms.ToPILImage(),
        transforms.RandomRotation(5, PIL.Image.BILINEAR),
        transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.Resize(32),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((args.mean_image,), (args.std_image,))])
    if not args.augmentation:
        transform_train = transform
        transform_test = transform

    train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         load_random_triplets=load_random_triplets,
                         batch_size=args.batch_size,
                         root=args.dataroot,
                         name=args.training_set,
                         download=True,
                         transform=transform_train),
        batch_size=args.batch_size,
        shuffle=False)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                         TripletPhotoTour(train=False,
                                          batch_size=args.test_batch_size,
                                          root=args.dataroot,
                                          name=name,
                                          download=True,
                                          transform=transform_test),
                         batch_size=args.test_batch_size,
                         shuffle=False)}
                    for name in test_dataset_names]

    return train_loader, test_loaders


def get_new_learning_rate():
    new_lr = args.lr * (1.0 - float(global_step) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    print("lr = {}".format(new_lr))
    return new_lr


def train(sess, saver, train_loader, train_op, loss_op, placeholders, epoch, load_triplets=False):
    """ A single epoch of the training data """
    global global_step

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:

        # Don't super like hard-coding shape here
        new_shape = (-1, 32, 32, 1)

        # Learning rate annealing
        learning_rate = get_new_learning_rate()

        if load_triplets:
            data_a, data_p, data_n = data

            data_a = data_a.numpy().reshape(new_shape)
            data_p = data_p.numpy().reshape(new_shape)
            data_n = data_n.numpy().reshape(new_shape)

            feed_dict = {
                placeholders.data_a: data_a,
                placeholders.data_p: data_p,
                placeholders.data_n: data_n,
                placeholders.learning_rate: learning_rate
            }

        else:
            data_a, data_p = data

            data_a = data_a.numpy().reshape(new_shape)
            data_p = data_p.numpy().reshape(new_shape)

            feed_dict = {
                placeholders.data_a: data_a,
                placeholders.data_p: data_p,
                placeholders.learning_rate: learning_rate
            }

        fetches = [train_op, loss_op]
        _, loss = sess.run(fetches, feed_dict)
        global_step += 1

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))

    try:
        os.stat('{}{}'.format(args.model_dir, suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir, suffix))

    saver.save(sess, conf.ckpt_dir + "HardNet/checkpoint_{}.ckpt".format(epoch))


def test(sess, test_loader, _out_a, _out_p, placeholders, epoch):
    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        new_shape = (-1, 32, 32, 1)
        data_a = data_a.numpy().reshape(new_shape)
        data_p = data_p.numpy().reshape(new_shape)

        feed_dict = {
            placeholders.data_a: data_a,
            placeholders.data_p: data_p
        }

        fetches = [_out_a, _out_p]
        out_a, out_p = sess.run(fetches, feed_dict)

        dists = np.sqrt(np.power(np.sum(out_a - out_p, axis=1), 2))  # euclidean distance
        distances.append(dists.reshape(-1, 1))
        ll = label.data.cpu().numpy().reshape(-1, 1)

        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    # distances = np.array([1.0 / (d + 1e-8) for d in distances])

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))


def HardNet_forward(model, input):
    """ Includes input and output norm stuff """
    input_norm = HardNet.start(input=input)
    x_features = model(input=input_norm)
    return HardNet.end(x_features)


class PlaceHolders:
    """ Container for all the placeholders """
    data_a = None
    data_p = None
    data_n = None
    learning_rate = None


def main(train_loader, test_loaders, model):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    with tf.variable_scope("input"):
        # aka gray scale patches
        # 3 inputs for the triplet loss
        # data_n may or may not be used (will be pruned from graph if not)
        placeholders = PlaceHolders()
        placeholders.data_a = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        placeholders.data_p = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        placeholders.data_n = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])

    # Output node
    out_a = HardNet_forward(model, input=placeholders.data_a)
    out_p = HardNet_forward(model, input=placeholders.data_p)
    out_n = HardNet_forward(model, input=placeholders.data_n)

    # print("out_a {}".format(out_a.get_shape().as_list()))

    # Loss node
    print("Batch reduce = {}".format(args.batch_reduce))
    if args.batch_reduce == 'L2Net':
        loss_op = loss_L2Net(out_a, out_p, anchor_swap=args.anchorswap,
                             margin=args.margin, loss_type=args.loss)
    elif args.batch_reduce == 'random_global':
        loss_op = loss_random_sampling(out_a, out_p, out_n,
                                       margin=args.margin,
                                       anchor_swap=args.anchorswap,
                                       loss_type=args.loss)
    else:
        loss_op = loss_HardNet(out_a, out_p,
                               margin=args.margin,
                               anchor_swap=args.anchorswap,
                               anchor_ave=args.anchorave,
                               batch_reduce=args.batch_reduce,
                               loss_type=args.loss)

    # Add extra loss terms
    if args.decor:
        loss_op += CorrelationPenaltyLoss(out_a)

    if args.gor:
        loss_op += args.alpha * global_orthogonal_regularization(out_a, out_n)

    # Add the regularisation terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    _lambda = 0.001
    loss_op += loss_op + _lambda * sum(reg_losses)

    # learning_rate = tf.train.exponential_decay(args.lr, global_step, 100000, 1e-6, staircase=True)
    placeholders.learning_rate = tf.placeholder(tf.float32, shape=[])

    # TODO: Check this is exactly the same as original HardNet optimizer (weight decay?)
    train_op = tf.train.MomentumOptimizer(learning_rate=placeholders.learning_rate, momentum=0.9).minimize(loss_op)

    # Save and restore variables
    saver = tf.train.Saver()

    # Initialising variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        num_params = 0
        for v in tf.trainable_variables():
            num_params += tfvar_size(v)

        print("Number of parameters = {}".format(num_params))

        # Initialize weights
        sess.run(init_op)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {}'.format(args.resume))
                saver.restore(sess, args.resume)
            else:
                print('=> no checkpoint found at {}'.format(args.resume))

        start = args.start_epoch
        end = start + args.epochs

        for epoch in range(start, end):

            # iterate over test loaders and test results
            train(sess, saver, train_loader, train_op, loss_op, placeholders, epoch, triplet_flag)

            for test_loader in test_loaders:
                test(sess, test_loader['dataloader'], out_a, out_p, placeholders, epoch)

            # randomize train loader batches
            train_loader, test_loaders2 = create_loaders(load_random_triplets=triplet_flag)


if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    if TEST_ON_W1BS:
        if not os.path.isdir(DESCS_DIR):
            os.makedirs(DESCS_DIR)

    # Create the architecture
    architecture = HardNet()
    # Use standard network
    #model = StandardNetwork(architecture=architecture)
    #model.build("StandardNetwork_HardNet")

    model = TuckerNet(architecture=architecture)
    model.build(conv_ranks, fc_ranks, "TuckerhardNet")

    train_loader, test_loaders = create_loaders(load_random_triplets=triplet_flag)
    main(train_loader, test_loaders, model)
