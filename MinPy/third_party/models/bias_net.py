import torch as t
import torch.nn.functional as F
from torch import nn
import numpy as np
class Linear(t.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = t.nn.Parameter(t.Tensor(output_features, input_features))
        if bias:
            self.bias = t.nn.Parameter(t.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    def forward(self, input):
        result = t.mm(input,self.weight.t())
        result = t.add(result,self.bias)
        return result

class bias_net(t.nn.Module):
    def __init__(self, para=[2,2000,1000,500,200,1],std_b=1e-1,bn_if=False,act='relu',std_w=1e-3):
        super(bias_net, self).__init__()
        self.bn_if = bn_if
        if act == 'relu':
            self.act = F.relu
        elif act == 'sigmoid':
            self.act = F.sigmoid
        elif act == 'tanh':
            self.act = F.tanh
        elif act == 'softmax':
            self.act = F.softmax
        elif act == 'threshold':
            self.act = F.threshold
        elif act == 'hardtanh':
            self.act = F.hardtanh
        elif act == 'elu':
            self.act = F.elu
        elif act == 'relu6':
            self.act = F.relu6
        elif act == 'leaky_relu':
            self.act = F.leaky_relu
        elif act == 'prelu':
            self.act = F.prelu
        elif act == 'rrelu':
            self.act = F.rrelu
        elif act == 'logsigmoid':
            self.act = F.logsigmoid
        elif act == 'hardshrink':
            self.act = F.hardshrink
        elif act == 'tanhshrink':
            self.act = F.tanhshrink
        elif act == 'softsign':
            self.act = F.softsign
        elif act == 'softplus':
            self.act = F.softplus
        elif act == 'softmin':
            self.act = F.softmin
        elif act == 'softmax':
            self.act = F.softmax
        elif act == 'log_softmax':
            self.act = F.log_softmax
        elif act == 'softshrink':
            self.act = F.softshrink
        elif act == 'sin':
            self.act = t.sin
        else:
            print('Wrong act name:',act)
        for i in range(len(para)-1):
            exec('self.fc'+str(i)+' = nn.Linear(para['+str(i)+'],para['+str(i+1)+'],bias=True)')
        if act == 'sin':
            for layer_i,m in enumerate(self.modules()):
                if isinstance(m, Linear):
                    m.weight.data = t.nn.init.uniform_(m.weight.data,-1 / para[layer_i]/30, 
                                             1 / para[layer_i]/30)
            self.fc0.weight.data = t.nn.init.uniform_(self.fc0.weight.data,-np.sqrt(6 / para[0]), 
                                             np.sqrt(6 / para[0]))
        else:
            for m in self.modules():
                if isinstance(m, Linear):
                    m.weight.data = t.nn.init.kaiming_normal_(m.weight.data)
                    m.bias.data = t.nn.init.constant_(m.bias, 0)
            self.fc0.weight.data = t.nn.init.normal_(self.fc0.weight.data,mean=0,std=std_w)
            self.fc0.bias.data = t.nn.init.normal_(self.fc0.bias.data,mean=0,std=std_b)
        for i in range(len(para)-1):
            exec('self.bn'+str(i)+' = t.nn.BatchNorm1d(para['+str(i+1)+'])')

    def forward(self, x):
        act_func = self.act
        if self.act == t.sin:
            x = act_func(30*self.fc0(x))
        else:
            x = act_func(self.fc0(x))
        if self.bn_if:
            x = self.bn0(x)
        if self.act == t.sin:
            x = act_func(30*self.fc1(x))
        else:
            x = act_func(self.fc1(x))
        if self.bn_if:
            x = self.bn1(x)
        if self.act == t.sin:
            x = act_func(30*self.fc2(x))
        else:
            x = act_func(self.fc2(x))
        if self.bn_if:
            x = self.bn2(x)
        if self.act == t.sin:
            x = act_func(30*self.fc3(x))
        else:
            x = act_func(self.fc3(x))
        if self.bn_if:
            x = self.bn3(x)
        x = self.fc4(x)
        return x