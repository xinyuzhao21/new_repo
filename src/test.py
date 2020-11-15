import data.dataset as DataSet
import data.datautil as util
from torch.utils.data import DataLoader
from torchvision import transforms
p_root = '../rendered_256x256/256x256/photo/tx_000000000000'
sk_root ='../rendered_256x256/256x256/sketch/tx_000000000000'

# https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory


# Test train_test_split
# util.train_test_split(p_root)
# util.train_test_split(sk_root)

# Test ImageDataSet
# tmp_root ='../test'
# util.train_test_split(tmp_root,split=(0.8,0.1,0.1))
# train_data = DataSet.ImageDataset(tmp_root,transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_data, shuffle=True)
#
# for x_batch, y_batch in train_loader:
#     print(x_batch.shape)
#     to_image = transforms.ToPILImage()
#     image =to_image(x_batch[0])
#     util.showImage(image)
#     print(train_data.class_to_idx)
#     print(y_batch)

# Test PairedDataSet
# tmp_root ='../testpair/photo'
# util.train_test_split(tmp_root,split=(0.8,0.1,0.1))
# sketch_root = '../testpair/sketch'
# train_data = DataSet.PairedDataset(photo_root=tmp_root,sketch_root=sketch_root,transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_data, batch_size=2,shuffle=True)
#
# for x_batch, y_batch in train_loader:
#     print(x_batch[0].shape,)
    # to_image = transforms.ToPILImage()
    # sketch,photo=x_batch
    # for i in range(sketch.shape[0]):
    #     image =to_image(sketch[i])
    #     util.showImage(image)
    #     image =to_image(photo[i])
    #     util.showImage(image)
    # print(train_data.class_to_index)
    # print(y_batch)

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


def main():
    # batch_size = 100
    batch_size = 2
    print("here")
    sk_root = '../256x256/sketch/tx_000000000000'
    sk_root = '../test'
    in_size = 225
    in_size = 224
    train_dataset = DataSet.ImageDataset(sk_root, transform=Compose([Resize(in_size), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)
    tmp_root ='../testpair/photo'
    # util.train_test_split(tmp_root,split=(0.8,0.1,0.1))
    sketch_root = '../testpair/sketch'
    train_dataset = DataSet.PairedDataset(photo_root=tmp_root,sketch_root=sketch_root,transform=Compose([Resize(in_size), ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)
    test_dataset = DataSet.PairedDataset(photo_root=tmp_root,sketch_root=sketch_root,transform=Compose([Resize(in_size), ToTensor()]),train=True)
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count(),
                                  shuffle=True, drop_last=True)

    model = SketchANet(num_classes=3)
    model = Net()
    crit = torch.nn.CrossEntropyLoss()
    net1 = getResnet(num_class=100,pretain=True)
    margin = 1
    model = SiameseNet(net1,net1)
    crit = ContrastiveLoss(margin)
    if torch.cuda.is_available():
        model = model.cuda()



    optim = torch.optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Tensorboard stuff
    # writer = tb.SummaryWriter('./logs')
    to_image = transforms.ToPILImage()
    count = 0
    epochs = 200
    prints_interval = 1
    for e in range(epochs):
        print('epoch', e, 'started')
        for i, (X, Y) in enumerate(train_dataloader):

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            sketch,photo = X
            Y = (Y != train_dataset.class_to_index['unmatched'])
            # for i in range(sketch.shape[0]):
            #     image =to_image(sketch[i])
            #     util.showImage(image)
            #     image =to_image(photo[i])
            #     util.showImage(image)
            #     print(Y)
            optim.zero_grad()
            #
            # image = to_image(X[0])
            # util.showImage(image)
            # print(train_dataset.class_to_idx)
            # print(Y)
            output = model(*X)
            # print(output,Y)

            loss = crit(*output, Y)

            if i % prints_interval == 0:
                print(f'[Training] {i}/{e}/{epochs} -> Loss: {loss.item()}')
                # writer.add_scalar('train-loss', loss.item(), count)

            # to_image = transforms.ToPILImage()
            # image = to_image(X[0])
            # util.showImage(image)
            # print(train_dataset.class_to_idx)
            # print(Y)

            loss.backward()
            optim.step()

            count += 1
        print('epoch', e, 'loss', loss.item())
        correct, total, accuracy = 0, 0, 0
        model.eval()
        # print(f'[Testing] -/{e}/{epochs} -> Accuracy: {accuracy} %', total, correct)
        model.train()

if __name__ == '__main__':
    main()
