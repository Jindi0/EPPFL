#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return torch.as_tensor(image), torch.as_tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, user_index, device):
        self.args = args
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), 
                    batch_size=self.args.local_bs, shuffle=True)                    
        self.user_index = user_index
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
       

    def update_weights(self, model, global_round):
        
        epoch_loss = []
        batchLoss = [] 
        train_acc = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            # Set mode to train model
            model.train()
            batch_loss = []
            total_images = 0
            correct_images = 0
            running_loss = 0
            tmp_acc = []

            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)
       
                optimizer.zero_grad()
                outputs = model(images)

                _, predicts = torch.max(outputs.data, 1)
                correct_images += (predicts == labels).sum().item()
                total_images += labels.size(0)

                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)

                optimizer.step()
                running_loss += loss.item()                

                if (batch_idx + 1) % 10 == 0:
                    batch_loss.append(loss.item())
                    tmp_acc.append(100 * correct_images / total_images)
                    
                    print('| Global Round : {} | Local : [{}, {}] | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.5f}'.format(
                        global_round, iter+1, batch_idx + 1,  total_images, len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), running_loss / 10, 100 * correct_images / total_images))
                    total_images = 0
                    correct_images = 0
                    running_loss = 0
                              
            batchLoss.append(batch_loss)
            train_acc.append(tmp_acc)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), batchLoss, train_acc 


    def inference(self, dataset, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.validloader):

            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        loss = loss / (batch_idx + 1)

        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
                            
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    loss = loss / total

    return accuracy, loss
