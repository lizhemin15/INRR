import torch as t
import torch.nn.functional as F

class fk_net(t.nn.Module):
    def __init__(self,x,act='relu'):
        super(fk_net, self).__init__()
        self.get_act(act)
        self.conv = t.nn.Conv2d(1, 1, kernel_size=(x.shape[-2],x.shape[-1]), stride=1, padding=0, bias=True)

    def get_act(self,act):
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
        else:
            print('Wrong act name:',act)

    def FK_block(self,x):
        x_pad = x.repeat(1,1,2,2)
        x_clip = x_pad[:,:,:-1,0:-1]
        x = self.conv(x_clip)
        return x

    def forward(self, x):
        #print(x.shape)
        for i in range(4):
            x = self.FK_block(x)
            #x = self.act(x)
        return x


