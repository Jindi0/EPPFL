#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid
import xlwt
import numpy as np


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './EPPFL/data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
    

def init_log(args):
    '''
    Initialize the log file
    '''
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('global',cell_overwrite_ok=True)
    row0 = ['General setting', 'FL setting', 
            'DP setting', 'Communication setting']

    general_args = 'task = {} \n \
            model = {} \n\
            dataset = {} \n\
            num_classes = {} \n\
            gpu = {} \n\
            iid = {} \n\
            unequal = {} \n\
            seed = {} \
            continue from = {}'.format(
                args.task, args.model, 
                args.dataset, args.num_classes,
                args.gpu, args.iid, args.unequal, 
                args.seed, args.contfrom)

    fl_args = 'globel round = {} \n\
            num_users = {} \n\
            frac = {} \n\
            local_epoch = {} \n\
            batch_size = {} \n\
            lr = {} \n\
            optimizer = {} \n\
            momentum = {}'.format(
                args.epochs, args.num_users, 
                args.frac, args.local_ep, args.local_bs, 
                args.lr, args.optimizer, args.momentum)

    dp_args = 'dp = {} \n\
            gama = {} \n\
            alpha = {} \n\
            u = {} \n\
            beta = {} \n\
            epsilon = {} \n\
            noise_dist_para = {}'.format(
                args.dp, args.gama, args.alpha, 
                args.u, args.beta, args.epsilon, 
                args.noise_dist_para)

    comm_args = 'reduced = {} \n\
            keep rate = {}'.format(
                args.reduced, args.keep_rate)

    row1 = [general_args, fl_args, dp_args, comm_args]

    style = xlwt.XFStyle()
    style.alignment.wrap = 1
    for i in range(len(row0)):
        sheet1.write(0,i, row0[i])
        sheet1.write(1, i, row1[i], style)

    acc_loss = ['Test acc', 'Train loss', 'Round time']

    for i in range(len(acc_loss)):
        sheet1.write(0, len(row1) + 2 + i, acc_loss[i])
    return f, sheet1




def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def merge_encrypt_models(public_key, x):
    x_en = copy.deepcopy(x[0])
    for key in x_en.keys(): 
        print('----------' + key)
        
        tmp = x[0][key].cpu().numpy()
        shape = tmp.shape
        tmp = tmp.flatten().tolist()
        tmp = [public_key.encrypt(i) * 1/len(x) for i in tmp]
        tmp = np.array(tmp)
        tmp = np.reshape(tmp, shape)
        sum_ = tmp
              
        for i in range(1, len(x)):
            tmp = x[i][key].cpu().numpy()
            shape = tmp.shape
            tmp = tmp.flatten().tolist()
            tmp = [public_key.encrypt(i) * 1/len(x) for i in tmp]
            tmp = np.array(tmp)
            tmp = np.reshape(tmp, shape)
            sum_ += tmp
        x_en[key] = sum_
    return x_en


# def decrypt_model(private_key, x):
#     x_en = copy.deepcopy(x)
#     for key in x_en.keys():
#         tmp = x[key]
#         shape = tmp.shape
#         tmp = tmp.flatten().tolist()
#         tmp = [private_key.decrypt(i) for i in tmp]
#         tmp = np.array(tmp, dtype=float)
#         tmp = np.reshape(tmp, shape)
#         tmp = torch.from_numpy(tmp)
#         x_en[key] = tmp
#     return x_en
