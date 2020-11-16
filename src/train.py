import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
from model.sketchanet import SketchANet,Net
import data.dataset as DataSet
from torch.utils.data import DataLoader
from model.resnet import getResnet
from torchvision import transforms
import data.datautil as util

def set_checkpoint(epoch,model,optimizer,train_loss,accurate,path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accurate': accurate,
        'train_loss':train_loss
    }, path)
    print("Checkpoint saved",'epoch',epoch,'train_loss',train_loss,'accurate',accurate)
def load_checkpoint(path,model,optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accurate = checkpoint['accurate']
    train_loss = checkpoint['train_loss']
    print("Checkpoint loaded",'epoch',epoch,'train_loss',train_loss,'accurate',accurate)



def main():
    # batch_size = 100
    batch_size = 1
    print("here")
    sk_root = '../256x256/sketch/tx_000000000000'
    sk_root = '../256x256/photo/tx_000000000000'
    sk_root ='../test_pair/sketch'
    in_size = 225
    in_size = 224
    train_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize(in_size), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)

    test_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize(in_size), ToTensor()]),train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)

    num_class = len(train_dataset.classes)
    embed_size = -1
    model = getResnet(num_class=num_class,embed_size=embed_size,pretrain=True)
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Tensorboard stuff
    # writer = tb.SummaryWriter('./logs')



    count = 0
    epochs = 200
    prints_interval = 1
    max_chpt = 3
    max_acu = -1
    chpt_num = 0
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    model.avgpool.register_forward_hook(get_activation('avgpool'))
    for e in range(epochs):
        print('epoch',e,'started')
        avg_loss = 0
        for i, (X, Y) in enumerate(train_dataloader):

            activation = {}




            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optim.zero_grad()
            to_image = transforms.ToPILImage()
            output = model(X)
            print(activation['avgpool'].shape)
            loss = crit(output, Y)
            avg_loss += loss.item()
            if i == 0:
                print(loss)
            if i % prints_interval == 0:
                print(f'[Training] {i}/{e}/{epochs} -> Loss: {avg_loss/(i+1)}')
                # writer.add_scalar('train-loss', loss.item(), count)
            loss.backward()

            optim.step()

            count += 1
        print('epoch',e,'loss',avg_loss/len(train_dataloader))
        correct, total, accuracy= 0, 0, 0
        # model.eval()
        for i, (X, Y) in enumerate(test_dataloader):

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            output = model(X)
            _, predicted = torch.max(output, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()


        accuracy = (correct / total) * 100

        print(f'[Testing] -/{e}/{epochs} -> Accuracy: {accuracy} %',total,correct)
        # model.train()
        if accuracy >= max_acu:
            path = 'checkpoint'+str(chpt_num)+'.pt'
            max_acu = accuracy
            chpt_num= (chpt_num+1)%max_chpt
            set_checkpoint(epoch=e,model=model,optimizer=optim,train_loss=avg_loss/len(train_dataloader),accurate=accuracy,path=path)
            path = 'best.pt'
            set_checkpoint(epoch=e,model=model,optimizer=optim,train_loss=avg_loss/len(train_dataloader),accurate=accuracy,path=path)
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
