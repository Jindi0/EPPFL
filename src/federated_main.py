#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import multiprocessing
from multiprocessing import Manager
import copy
import time
import numpy as np
from tqdm import tqdm
import xlwt
import torch
from options import args_parser
from update import LocalUpdate, test_inference
from models import *
from utils import *
from collections import OrderedDict
from xlutils.copy import copy as xlcopy
from xlrd import open_workbook




def client(i, dic, args, train_dataset, user_images, idx, global_model, epoch):
    # set device
    device_id = (i + 2) % 3
    torch.cuda.set_device(device_id)
    device = 'cuda' 
    global_model.to(device)
    print('client id = ' + str(i))

    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_images,  
                                    user_index=idx, device=device)

    w, loss, batchLoss, train_acc = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

    tmp = {}
    tmp_w =  OrderedDict()

    for key, values in w.items():
        tmp_w[key] = values.to('cpu')

    tmp['local_weights'] = tmp_w
    tmp['local_losses'] = loss

    dic[str(i)] = tmp

    


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = args_parser()

    f_xls, sheet1 = init_log(args)

    start_time1 = time.time()
    

    # create directories
    save_path = './EPPFL/save/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    folder_name = args.task
    model_path = './EPPFL/save/{}/'.format(folder_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ckp_path = model_path + 'ckps/'
    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)

    # set device
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # build model
    if args.model == 'VGG19':
        global_model = VGG19(args=args)
    else:
        exit('Error: unrecognized model')


    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []

    print_every = 10
    val_loss_pre, counter = 0, 0
    iter_num = 1
    
    # Continue trining from a checkpoint
    if args.contfrom != 0:
        global_model = torch.load(ckp_path + str(args.startfrom))
        global_weights = global_model.state_dict()
        f_xls = open_workbook('./EPPFL/save/{}/{}.xls'.format(folder_name, folder_name),formatting_info=True)
        f_xls = xlcopy(f_xls) 
        sheet1 = f_xls.get_sheet(0)

    
    for epoch in tqdm(range(args.epochs)):
        epoch += args.contfrom + 1
        epoch_start_time = time.time()
        local_weights, local_losses = [], []

        print(f'\n | Global Training Round : {epoch} |\n')

        global_model.train()
        num_active_users = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), num_active_users, replace=False)

        process_list = []
        dic = Manager().dict()

        # training local models
        for i in range(0, num_active_users):
            idx = idxs_users[i]
            user_images = user_groups[idx]

            p = multiprocessing.Process(target=client, args=(i, dic, args, 
                            train_dataset, user_images, idx, global_model, epoch))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        for p in process_list:
            p.terminate()
    
        # update global weights
        for key, item in dic.items():
            tmp = OrderedDict()
            for innerkey, value in item['local_weights'].items():
                tmp[innerkey] = value.to(device)

            local_weights.append(tmp)
            local_losses.append(item['local_losses'])
            
        # merge models
        global_weights = average_weights(local_weights)


        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)  # global training loss
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        for idx in range(1):
            global_model.to(device)
            acc, _ = test_inference(args=args, test_dataset=test_dataset,model=copy.deepcopy(global_model))
            global_model.to('cpu')
            list_acc.append(acc)
           
        test_accuracy.append(sum(list_acc)/len(list_acc))
        avg_acc = sum(list_acc)/len(list_acc)

        sheet1.write(1+epoch, int(6) ,avg_acc)
        sheet1.write(1+epoch, int(7) ,sum(train_loss)/len(train_loss))
        sheet1.write(1+epoch, int(8) ,time.time() - epoch_start_time)
        f_xls.save('./EPPFL/save/{}/{}.xls'.format(folder_name, folder_name))
        

        if (epoch+1) % print_every == 0:
            torch.save(global_model, ckp_path + str(epoch))
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*test_accuracy[-1]))
            print('\n Run Time: {0:0.4f}'.format(time.time()-epoch_start_time))

        train_loss = []

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*acc))
    print('|---- Total Run Time: {0:0.4f}'.format(time.time()-start_time1))

    f_xls.save('./EPPFL/save/{}/{}.xls'.format(folder_name, folder_name))





        

        

        
        

    