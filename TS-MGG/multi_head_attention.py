import torch

import os
import dgl
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = torch.sum(attention, dim=-1)
        attention = torch.unsqueeze(F.softmax(attention, dim=-1), dim=1)

        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)  # (node_num, )

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out


if __name__ == '__main__':
    # d_tensor = torch.randn((10, 5))
    # t_tensor = torch.randn((10, 5))
    # s_tensor = torch.randn((10, 5))
    # attention = torch.randn((10, 3))
    # s_attention = torch.softmax(attention, dim=-1)
    # print(s_attention)

    torch.manual_seed(0)

    d_tensor = torch.arange(start=0, end=30, step=1, dtype=torch.float32).reshape((3, 10))
    t_tensor = torch.arange(start=30, end=60, step=1, dtype=torch.float32).reshape((3, 10))
    s_tensor = torch.arange(start=60, end=90, step=1, dtype=torch.float32).reshape((3, 10))

    total_tensor = torch.cat((d_tensor, t_tensor, s_tensor), dim=1).reshape(3, 3, 10)
    # result = torch.matmul(total_tensor, total_tensor.permute(0, 2, 1))
    # value = torch.sum(result, dim=-1)
    # value_2 = torch.sum(result, dim=-2)
    # print(value)
    # print(value_2)
    # print(value.size())
    # print(value_2.size())

    # attention = torch.tensor([[[0.0, 0.5, 0.5]],
    #                           [[0.5, 0.5, 0.0]],
    #                           [[0.5, 0.0, 0.5]]], dtype=torch.float32)
    # attention = torch.unsqueeze(attention, dim=1)
    # print(total_tensor)
    # print(torch.matmul(attention, total_tensor))

    attention = Multi_Head_Attention(10, 2, 0.5)
    out = attention(total_tensor)
    out = torch.squeeze(out)
    print(out.size())
