import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
from src.model.sketchanet import SketchANet,Net
import src.data.dataset as DataSet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import src.data.datautil as util
import torchvision
def main( ):
    # quickdraw_trainds = ImageFolder(args.traindata, transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()]))
    # train_dataloader = DataLoader(quickdraw_trainds, batch_size=args.batch_size, pin_memory=True, num_workers=os.cpu_count(),
    #     shuffle=True, drop_last=True)
    #
    # quickdraw_testds = ImageFolder(args.testdata, transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()]))
    # test_dataloader = DataLoader(quickdraw_testds, batch_size=args.batch_size * 2, pin_memory=True, num_workers=os.cpu_count(),
    #     shuffle=True, drop_last=True)
    batch_size = 200
    batch_size = 4
    print("here")
    sk_root = '../rendered_256x256/256x256/sketch/tx_000000000000'
    # sk_root ='../test'
    train_dataset = ImageFolder(sk_root, transform=Compose([Resize([225, 225]), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)

    # transform = transforms.Compose(
    #     [Resize([225, 225]),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    test_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize([225, 225]), ToTensor()]),train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                         shuffle=True, drop_last=True)


    model = SketchANet(num_classes=125)
    # model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Tensorboard stuff
    # writer = tb.SummaryWriter('./logs')

    count = 0
    epochs = 20
    prints_interval = 1000
    for e in range(epochs):
        for i, (X, Y) in enumerate(train_dataloader):

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optim.zero_grad()
            # to_image = transforms.ToPILImage()
            # image = to_image(X[0])
            # util.showImage(image)
            # print(train_dataset.class_to_idx)
            # print(Y)
            output = model(X)
            loss = crit(output, Y)
            
            if i % prints_interval == 0:
                print(f'[Training] {i}/{e}/{epochs} -> Loss: {loss.item()}')
                # writer.add_scalar('train-loss', loss.item(), count)
            
            loss.backward()
            optim.step()

            count += 1

        # correct, total, accuracy= 0, 0, 0
        # for i, (X, Y) in enumerate(test_dataloader):
        #     # Binarizing 'X'
        #     X[X < 1.] = 0.; X = 1. - X
        #
        #     if torch.cuda.is_available():
        #         X, Y = X.cuda(), Y.cuda()
        #     output = model(X)
        #     _, predicted = torch.max(output, 1)
        #     total += Y.size(0)
        #     correct += (predicted == Y).sum().item()
        #     accuracy = (correct / total) * 100
        #
        # print(f'[Testing] -/{e}/{epochs} -> Accuracy: {accuracy} %')
        # writer.add_scalar('test-accuracy', accuracy/100., e)

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