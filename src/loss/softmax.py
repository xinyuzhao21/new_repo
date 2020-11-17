import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMax(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, embed_size,num_class):
        super(SoftMax, self).__init__()
        self.fc = nn.Linear(embed_size,num_class)
        self.crit = nn.CrossEntropyLoss()

    def forward(self, x,y):
        logits = self.fc(x)
        loss = self.crit(logits,y)
        return logits,loss