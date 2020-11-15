# https://github.com/adambielski/siamese-triplet/blob/master/networks.py
import torch
from  torch import  nn

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
