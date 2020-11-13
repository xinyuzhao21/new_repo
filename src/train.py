import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
from model.sketchanet import SketchANet,Net
import data.dataset as DataSet
from torch.utils.data import DataLoader

from torchvision import transforms
import data.datautil as util

def set_checkpoint(epoch,model,optimizer,metric,loss,path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path,model,optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Checkpoint loaded",'epoch',epoch,'loss',loss)



def main():
    # batch_size = 100
    batch_size = 1
    print("here")
    sk_root = '../256x256/sketch/tx_000000000000'
    sk_root ='../test'
    train_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize(225), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)

    test_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize([225, 225]), ToTensor()]),train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)


    model = SketchANet(num_classes=3)
    model = Net()
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
    for e in range(epochs):
        print('epoch',e,'started')
        for i, (X, Y) in enumerate(train_dataloader):

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optim.zero_grad()
            to_image = transforms.ToPILImage()
            image = to_image(X[0])
            util.showImage(image)
            print(train_dataset.class_to_idx)
            print(Y)
            output = model(X)
            #print(output,Y)
            loss = crit(output, Y)
            
            if i % prints_interval == 0:
                print(f'[Training] {i}/{e}/{epochs} -> Loss: {loss.item()}')
                # writer.add_scalar('train-loss', loss.item(), count)
            loss.backward()

            optim.step()

            count += 1
        print('epoch',e,'loss',loss.item())
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
