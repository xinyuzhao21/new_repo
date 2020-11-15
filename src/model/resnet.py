import torchvision.models as models
import torch.nn as nn
def getResnet(num_class = 125, pretrain=False):
    model = models.resnet18(pretrained=pretrain)
    # for name, child in model.named_children():
    #     for name2, params in child.named_parameters():
    #         print(name, name2)
    model.fc = nn.Linear(model.fc.in_features,num_class)
    return model

model = getResnet()
