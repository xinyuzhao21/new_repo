# https://github.com/adambielski/siamese-triplet/blob/master/networks.py
import torch
from  torch import  nn
from model.linear import ClassificationNet
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiaClassNet(torch.nn.Module):
    def __init__(self, embedding_net,embedding_size,num_class):
        super(SiaClassNet, self).__init__()
        self.siamese = SiameseNet(embedding_net)
        self.fc = ClassificationNet(embedding_size,num_class)

    def forward(self, x1, x2):
        output1, output2 = self.siamese(x1,x2)
        out = self.fc((output2+output1)/2)
        return out

    def get_embedding(self, x, net1=True):
        return self.net1(x) if net1 else self.net2(x)