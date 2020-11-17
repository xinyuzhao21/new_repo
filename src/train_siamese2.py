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

def set_checkpoint(epoch, model, optimizer, train_loss, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_loss':train_loss
    }, path)
    print("saved epoch",epoch)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_loss=checkpoint['train_loss']
    print("Checkpoint loaded", 'epoch', epoch, 'loss', loss)



def main():
    # batch_size = 100
    batch_size = 100
    balanced = False
    print("Start Training")
    sk_root = '../256x256/sketch/tx_000000000000'
    # sk_root ='../test'
    in_size = 225
    in_size = 224
    tmp_root ='../test_pair/photo'
    sketch_root = '../test_pair/sketch'
    tmp_root = '../256x256/photo/tx_000000000000'
    sketch_root = '../256x256/sketch/tx_000000000000'
    train_dataset = DataSet.PairedDataset(photo_root=tmp_root,sketch_root=sketch_root,transform=Compose([Resize(in_size), ToTensor()]),balanced=balanced)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)
    test_dataset = DataSet.PairedDataset(photo_root=tmp_root,sketch_root=sketch_root,transform=Compose([Resize(in_size), ToTensor()]),train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)



    num_class = len(train_dataset.classes)
    embedding_size = 10242
    embedding_size = 1024
    embedding_size = 512
    net1 = getResnet(num_class=embedding_size, pretrain=True)
    margin = 1
    model = SiameseNet(net1)


    method = 'metric'
    crit = ContrastiveLoss(margin)
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Tensorboard stuff
    # writer = tb.SummaryWriter('./logs')

    count = 0
    epochs = 100
    prints_interval = 100
    prints_interval = 100
    max_chpt = 3
    min_loss = 100000
    chpt_num = 0
    for e in range(epochs):
        print('epoch', e, 'started')
        avg_loss = 0
        for i, (X, Y) in enumerate(train_dataloader):
            one = torch.ones(Y[0].shape)
            zero = torch.zeros(Y[0].shape)
            if torch.cuda.is_available():
                X, Y = (X[0].cuda(), X[1].cuda()), (Y[0].cuda(), Y[1].cuda(), Y[2].cuda())
                one,zero = one.cuda(),zero.cuda()
            output = model(*X)
            # print(output,Y)
            sketch,photo = X
            #print(sketch.shape)
            optim.zero_grad()
            to_image = transforms.ToPILImage()
            output = model(*X)
            #print(output[0])
            (Y,label_s,label_p) = Y
            Y = torch.where(Y != train_dataset.class_to_index['unmatched'], one, zero)
            loss = crit(*output,Y)
            avg_loss+=loss.item()
            if i % prints_interval == 0:
                # print(output,Y)
                print(f'[Training] {i}/{e}/{epochs} -> Loss: {avg_loss/(i+1)}')
                # writer.add_scalar('train-loss', loss.item(), count)
            loss.backward()

            optim.step()

            count += 1
        print('epoch', e, 'Avg loss', avg_loss/len(train_dataloader))
        valid_loss = eval_loss(test_dataloader,model,e,epochs,crit,train_dataset)
        if valid_loss< min_loss:
            path = 'checkpoint'+str(chpt_num)+'.pt'
            min_loss = valid_loss
            chpt_num= (chpt_num+1)%max_chpt
            set_checkpoint(epoch=e,model=model,optimizer=optim,train_loss=avg_loss/len(train_dataloader),loss=valid_loss,path=path)
            path = 'best.pt'
            set_checkpoint(epoch=e, model=model, optimizer=optim, train_loss=avg_loss / len(train_dataloader),
                           loss=valid_loss, path=path)

    # train_dataset = DataSet.ImageDataset(sketch_root, transform=Compose([Resize(in_size), ToTensor()]))
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
    #                      shuffle=True, drop_last=True)
    # test_dataset = DataSet.ImageDataset(sketch_root, transform=Compose([Resize(in_size), ToTensor()]),train=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
    #                      shuffle=True, drop_last=True)
    # train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_dataloader, model)
    # plot_embeddings(train_embeddings_baseline, train_labels_baseline,train_dataset)
    # val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_dataloader, model)
    # plot_embeddings(val_embeddings_baseline, val_labels_baseline,train_dataset)

def eval_loss(test_dataloader,model,e,epochs,crit,train_dataset):
    avg_loss = 0
    for i, (X, Y) in enumerate(test_dataloader):
        one = torch.ones(Y[0].shape)
        zero = torch.zeros(Y[0].shape)
 
        if torch.cuda.is_available():
            X, Y = (X[0].cuda(), X[1].cuda()), (Y[0].cuda(), Y[1].cuda(), Y[2].cuda)
            one,zero = one.cuda(),zero.cuda()
        to_image = transforms.ToPILImage()
        (Y, label_s, label_p) = Y
        to_image = transforms.ToPILImage()
        Y = torch.where(Y != train_dataset.class_to_index['unmatched'], one, zero)
        output = model(*X)

        # print(output,Y)
        loss = crit(*output, Y)
        avg_loss += loss.item()
    print(f'[Testing] -/{e}/{epochs} -> Avg Loss: {avg_loss / len(test_dataloader)} %')
    return avg_loss/len(test_dataloader)


def plot_embeddings(embeddings, targets,data, xlim=None, ylim=None):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    plt.figure(figsize=(10,10))
    for i in range(3):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(data.classes)
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()

            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

if __name__ == '__main__':
    #     import argparse
    #
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--traindata', type=str, required=True, help='root of the train datasets')
    #     parser.add_argument('--testdata', type=str, required=True, help='root of the test datasets')
    #     parser.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='Batch size')
    #     parser.add_argument('-c', '--num_classes', type=int, required=False, default=10, help='Number of classes for the classification task')
    #     parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='No. of epochs')
    #     parser.add_argument('-i', '--print_interval', type=int, required=False, default=10, help='Print loss after this many iterations')
    #     args = parser.parse_args()
    #
    main()
