# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:19:26 2018

@author: Kenji
"""


import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        
        return G.div(a * b * c * d)