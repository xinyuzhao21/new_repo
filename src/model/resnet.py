import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
def getResnet(num_class = 125, embed_size=-1, pretrain=False, feature_extract=False):
    model = models.resnet18(pretrained=pretrain)
    # for name, child in model.named_children():
    #     for name2, params in child.named_parameters():
    #         print(name, name2)
    # if embed_size <0:
    #     model.fc = nn.Linear(model.fc.in_features,num_class)
    # else:
    #     model.fc = HelperNet(num_class,embed_size,model.fc.in_features)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    if feature_extract:
        model.fc = Identity()
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class HelperNet(nn.Module):
    def __init__(self, num_class, embed_size,in_features):
        super(HelperNet, self).__init__()
        self.embed=nn.Linear(in_features, embed_size)
        self.classify=nn.Linear(embed_size, num_class)

    def forward(self, x):
        x = self.embed(x)
        x = x/x.pow(2).sum(1, keepdim=True).sqrt()
        x = F.relu(x)
        x = self.classify(x)
        return x

    def get_embedding(self, x):
        embed = self.embed(x)
        return embed/embed.pow(2).sum(1, keepdim=True).sqrt()
