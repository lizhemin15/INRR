import torch as t
import torch.nn.functional as F
from torch import nn
import numpy as np



class att_net(t.nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    def __init__(self,input_dim,dim_k):
        super(att_net, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, qx, kx):
        Q = self.q(qx) # Q: batch_size * seq_q_len * dim_k
        K = self.k(kx) # K: batch_size * seq_k_len * dim_k
        atten = nn.Softmax(dim=-1)(t.bmm(Q,K.permute(0,2,1))* self._norm_fact) # Q * K.T() # batch_size * seq_q_len * seq_k_len
        # TODO Performer random feature加速计算
        return atten