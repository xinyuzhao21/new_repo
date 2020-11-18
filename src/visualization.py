import torch
# helper function
from torch.utils.tensorboard import SummaryWriter
import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
from model.sketchanet import SketchANet,Net
import data.dataset as DataSet
from torch.utils.data import DataLoader
from model.resnet import getResnet
from torchvision import transforms
import data.datautil as util
import matplotlib.pyplot as plt
import numpy as np
import  torchvision
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter()
sk_root = '../rendered_256x256/256x256/sketch/tx_000000000000'

train_dataset = torchvision.datasets.ImageFolder(sk_root, transform=Compose([Resize(224), ToTensor()]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                        shuffle=True)

def set_checkpoint(epoch, model, softmax ,optimizer, train_loss, softmax_loss, hinge_loss,
                           accurate, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'softmax':softmax.state_dict(),
        'softmax_loss':softmax_loss,
        'hinge_loss':hinge_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'accurate': accurate,
        'train_loss': train_loss
    }, path)
    print("Checkpoint saved", 'epoch', epoch, 'train_loss', train_loss, 'accurate', accurate)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accurate = checkpoint['accurate']
    train_loss = checkpoint['train_loss']
    print("Checkpoint loaded", 'epoch', epoch, 'train_loss', train_loss, 'accurate', accurate)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


features= torch.load('features.pt')
print(features.shape)
sys.end()
dataiter = iter(train_dataloader)
images, labels = dataiter.next()
num_class = len(train_dataset.classes)
embed_size = -1
model = getResnet(num_class=3, pretrain=True)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
load_checkpoint('best.pt',model,optim)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100,
                                        shuffle=True)
dataiter = iter(train_dataloader)
images,labels = dataiter.next()
print(images.shape)
# get the class labels for each image
class_labels = [train_dataset.classes[lab] for lab in labels]
print(class_labels)
# log embeddings
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook
model.avgpool.register_forward_hook(get_activation('avgpool'))
_ = model(images)
features = activation['avgpool']
features = features.view(-1,512)
torch.save(features,'features.pt')
print(images.shape)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.view(-1,3,224,224))
writer.close()

