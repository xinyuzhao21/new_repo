# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
import torch
import torchvision
from torchvision import transforms
class PairedDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,sketch_root,photo_root,labels=None):
        'Initialization'
        self.labels = labels
        self.sketch_root = sketch_root
        self.photo_path = photo_root



  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

class ImageDataset(torchvision.datasets.ImageFolder):
    def __init__(self, image_root, transform=None, labels=None,train=True):
        self.image_root = image_root
        self.train_paths = []
        self.test_paths = []
        self.train = train
        train_links = self.image_root+'/train.txt'
        test_links = self.image_root + '/test.txt'
        with open(train_links) as f:
            for line in f:
                self.train_paths.append(line.split()[1])
        self.transform = transform
        with open(test_links) as f:
            for line in f:
                self.test_paths.append(line.split()[1])
        super(ImageDataset, self).__init__(image_root,is_valid_file=self.is_valid,transform=self.transform)

    def is_valid(self,path):
        return (not (path in self.test_paths) ) == self.train
