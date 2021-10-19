#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
from pickle import TRUE


def args_parser():
    parser = argparse.ArgumentParser()

    # general setting
    parser.add_argument('--task', type=str, default='eppfl', help='task name')
    parser.add_argument('--model', type=str, default='VGG19', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=1, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--contfrom', type=int, default=0, help='continue training from a checkpoint')
    
    # federated learning setting
    parser.add_argument('--epochs', type=int, default=2, help="the number of global rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="the number of users")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients')
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs in each global round")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # Differential privacy setting
    parser.add_argument('--dp', type=bool, default=True, help='if apply dp to activations')
    parser.add_argument('--gama', type=int, default=2, help='gama in normolization')
    parser.add_argument('--alpha', type=int, default=1, help='alpha in normolization')
    parser.add_argument('--u', type=int, default=4, help='u in normolization')
    parser.add_argument('--beta', type=float, default=0.5, help='beta in normolization')
    parser.add_argument('--epsilon', type=float, default=1, help='prvacy budget in dp')
    parser.add_argument('--noise_dist_para', type=float, default=0, \
        help='the parameter used in random noise generating function') # the average of parameters for images

    # communication cost setting
    parser.add_argument('--reduced', type=bool, default=False, help='if reduce transported data size')
    parser.add_argument('--keep_rate', type=float, default=0.5, help='the fraction of transferring data')
 
    args = parser.parse_args()
    return args
