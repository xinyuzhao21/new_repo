import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.nn import functional as F
from loss.contrastive import ContrastiveLoss
from loss.softmax import SoftMax
from model.sketchanet import SketchANet, Net
import data.dataset as DataSet
from torch.utils.data import DataLoader
from model.resnet import getResnet
from torchvision import transforms
import data.datautil as util


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


def main():
    batch_size = 100
    balanced = False
    print("Start Training")

    # sk_root ='../test'
    in_size = 225
    in_size = 224
    tmp_root = '../test_pair/photo'
    sketch_root = '../test_pair/sketch'
    tmp_root = '../256x256/photo/tx_000000000000'
    sketch_root = '../256x256/sketch/tx_000000000000'

    train_dataset = DataSet.PairedDataset(photo_root=tmp_root, sketch_root=sketch_root,
                                          transform=Compose([Resize(in_size), ToTensor()]), balanced=balanced)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)
    test_dataset = DataSet.PairedDataset(photo_root=tmp_root, sketch_root=sketch_root,
                                         transform=Compose([Resize(in_size), ToTensor()]), train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                 shuffle=True, drop_last=True)

    num_class = len(train_dataset.classes)
    embed_size = -1
    sketch_net = getResnet(num_class=num_class, pretrain=True,feature_extract=True)
    softmax_loss = SoftMax(embed_size=512, num_class=num_class)
    hinge_loss = ContrastiveLoss(margin=2)
    optim = torch.optim.Adam(list(sketch_net.parameters()) + list(softmax_loss.parameters()))
    sketch_net.train()
    photo_net = getResnet(num_class=num_class, pretrain=True, feature_extract=True)
    for param in photo_net.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        sketch_net = sketch_net.cuda()
        softmax_loss =softmax_loss.cuda()
        photo_net=photo_net.cuda()
    count = 0
    epochs = 200
    max_chpt = 3
    max_acu = -1
    chpt_num = 0
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook


    for e in range(epochs):
        print('epoch', e, 'Start')
        (avg_loss,avg_class_loss,avg_hinge_loss,accuracy)=eval_model(e,epochs,sketch_net,photo_net,softmax_loss,hinge_loss,[train_dataloader,test_dataloader],optim,train=True)
        print('epoch', e, 'End')
        (avg_loss,avg_class_loss,avg_hinge_loss,accuracy)=eval_model(e, epochs, sketch_net, photo_net, softmax_loss, hinge_loss, [train_dataloader, test_dataloader], optim,
                   train=False)

        if accuracy >= max_acu:
            path = 'checkpoint' + str(chpt_num) + '.pt'
            max_acu = accuracy
            chpt_num = (chpt_num + 1) % max_chpt
            set_checkpoint(epoch=e, model=sketch_net, softmax = softmax_loss,optimizer=optim, train_loss= avg_loss / len(train_dataloader), softmax_loss=avg_class_loss, hinge_loss=avg_hinge_loss,
                           accurate=accuracy, path=path)
            path = 'best.pt'
            set_checkpoint(epoch=e, model=sketch_net, softmax=softmax_loss, optimizer=optim,
                           train_loss=avg_loss / len(train_dataloader), softmax_loss=avg_class_loss,
                           hinge_loss=avg_hinge_loss,
                           accurate=accuracy, path=path)

def eval_model(e,epochs,sketch_net,photo_net,softmax_loss, hinge_loss, dataloaders,optim,train=True,prints_interval=100):
    avg_loss, avg_class_loss, avg_hinge_loss = 0, 0, 0
    correct, total, accuracy = 0, 0, 0
    if not train:
        sketch_net.eval()
        softmax_loss.eval()
        dataloader = dataloaders[1]
    else:
        sketch_net.train()
        softmax_loss.train()
        dataloader = dataloaders[0]

    for i, (X, Y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X, Y = (X[0].cuda(), X[1].cuda()), (Y[0].cuda(), Y[1].cuda(), Y[2].cuda())
        if train:
            optim.zero_grad()
        sketch, photo = X
        (Y, label_s, label_p) = Y
        to_image = transforms.ToPILImage()
        embeding_sketch = sketch_net(sketch)
        logits,class_loss = softmax_loss(embeding_sketch, label_s)
        avg_class_loss += class_loss.item()


        _, predicted = torch.max(logits, 1)
        total += Y.size(0)
        correct += (predicted == label_s).sum().item()

        embeding_photo = photo_net(photo)
        embeding_sketch = F.normalize(embeding_sketch, p=2, dim=1)
        embeding_photo = F.normalize(embeding_photo, p=2, dim=1)

        contrastive_loss = hinge_loss(embeding_sketch, embeding_photo, Y)
        avg_hinge_loss += contrastive_loss.item()
        loss = contrastive_loss + class_loss
        avg_loss += loss.item()
        if i % prints_interval == 0 and train:
            print(
                    f'[Training] {i}/{e}/{epochs} -> Acuracy: {correct/total}Loss: {avg_loss / (i + 1)} hinge loss: {avg_hinge_loss / (i + 1)} NLL: {avg_class_loss / (i + 1)}')
            # writer.add_scalar('train-loss', loss.item(), count)
        if train:
            loss.backward()
            optim.step()
    
    accuracy = (correct / (total+0.5)) * 100
    N=len(dataloader)
    avg_loss/=N
    avg_class_loss/=N
    avg_hinge_loss/=N
    if not train:
        print(f'[Testing] {N}/{e}/{epochs} -> Accuracy: {accuracy} Loss: {avg_loss} hinge loss: {avg_hinge_loss} NLL: {avg_class_loss}')
    else:
        print(f'[Training] {N}/{e}/{epochs} -> Loss: {avg_loss} hinge loss: {avg_hinge_loss} NLL: {avg_class_loss}')
    return (avg_loss,avg_class_loss,avg_hinge_loss,accuracy)

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
