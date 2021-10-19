#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import zlib
import random
import math
import json
import math

class VGG19(nn.Module):
    def __init__(self, args):
        super(VGG19, self).__init__()
        self.args = args
        self.gama = args.gama
        self.alpha = args.alpha
        self.u = args.u
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.l = 0 
        self.keep_rate = args.keep_rate

        self.clientlayers = nn.Sequential(
            # Conv 1-1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.serverlayers = nn.Sequential(
            # Conv 1-2

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv 2-1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 2-2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 3-1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 3-2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 3-3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 3-4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 4-1
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 4-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 4-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 4-4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 5-1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 5-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 5-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # Conv 5-4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),

            # # nn.MaxPool2d(kernel_size=2, stride=2),
            
        )

        self.classifier = nn.Sequential(
            # FC1
            nn.Linear(2048, 512), 
            nn.ReLU(True),
            nn.Dropout(),
            # FC2
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(), 
            # FC3
            nn.Linear(512, args.num_classes),   
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.normal_(1, 5)
                m.bias.data.zero_()

    def forward(self, x):
        # client-side model
        x = self.clientlayers(x)

        # reduce communication cost 
        if self.args.reduced:
            x = self.reduce_activations(x)
            x.register_hook(self.hook_fn)           

        # Differential Privacy 
        if self.args.dp: 
            x = self.add_noise(x)

        # server-side model
        x = self.serverlayers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    
    def add_noise(self, acts):
        n, m, w, h = list(acts.size())  # m = # kernels
        acts = [self.add_noise_channel(channel) for sample in acts for channel in sample ]
        acts = torch.stack(acts, dim=0)
        acts = acts.view(n, m, w, h)
        return acts
        
    
    def add_noise_channel(self, channel):
        w, h = list(channel.size())
        max_value = torch.max(channel)
        lap_param = max_value / self.epsilon  # sensitivity (scale)
        tmp = np.random.laplace(0, lap_param.item(), w*h)
        noise = torch.from_numpy(tmp).to(device='cuda', dtype=torch.float)
        noise = torch.reshape(noise, (w, h))
        return channel+noise

    def reduce_activations(self, x):
        # randomly discard some activations
        tmp = x.view(1, -1)
        activation_size = tmp.shape[1]
        discard_size = int(activation_size * (1 - self.keep_rate))
        keep_size = activation_size - discard_size
        
        ones = torch.ones([1, keep_size])
        zeros = torch.zeros([1, discard_size])
        random_vector = torch.cat((ones, zeros), dim=1).to(device='cuda')
        ind = torch.randperm(activation_size)
        random_vector = random_vector[0, ind]

        result_vector = tmp.mul(random_vector)
        result_vector = result_vector.view(x.shape)
        
        return result_vector


    def hook_fn(self, grad):
        # discard gradients that have small absolute values
        # print('--------hook function----------')

        tmp = grad.view(1, -1).to(device='cuda', dtype=torch.float)
        tmp = torch.abs(tmp)
        tmp, _ = torch.sort(tmp, descending=True)

        position = int(tmp.shape[1]*(self.keep_rate))
        split_value = tmp[0, position] 

        absresult = torch.abs(grad) < split_value
        grad = absresult.mul(grad)

        # print(grad)

        return grad

