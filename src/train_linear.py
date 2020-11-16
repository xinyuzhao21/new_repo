import torch, os
from loss.contrastive import ContrastiveLoss
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
# from torch.utils import tensorboard as tb
from model.sketchanet import SketchANet, Net
from model.siamesenet import SiameseNet
from model.resnet import getResnet
import data.dataset as DataSet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import data.datautil as util
import torchvision
from model.linear import ClassificationNet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_loss=checkpoint['train_loss']
    print("Checkpoint loaded", 'epoch', epoch, 'loss', loss)


embedding_size = 2
net1 = getResnet(num_class=embedding_size,pretrain=True)
margin = 10
model = SiameseNet(net1)
path = 'best.pt'
optim = torch.optim.Adam(model.parameters())
load_checkpoint(path,model,optim)