# https://github.com/adambielski/siamese-triplet/blob/master/networks.py
import torch
from  torch import  nn
from model.linear import ClassificationNet
class SiameseNet(torch.nn.Module):
    def __init__(self, net1,net2):
        super(SiameseNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self,x1,x2):
        output1 = self.net1(x1)
        output2 = self.net2(x2)
        return output1, output2

    def get_embedding(self, x, net1 = True):
        return self.net1(x) if net1 else self.net2(x)

class SiaClassNet(torch.nn.Module):
    def __init__(self, net1, net2,embedding_size,num_class):
        super(SiaClassNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.fc = ClassificationNet(embedding_size,num_class)

    def forward(self, x1, x2):
        output1 = self.net1(x1)
        output2 = self.net2(x2)
        out = self.fc((output2+output1)/2)
        return out

    def get_embedding(self, x, net1=True):
        return self.net1(x) if net1 else self.net2(x)