# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
# https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms

class ImageDataset(torchvision.datasets.ImageFolder):
    def __init__(self, image_root, transform=None, labels=None,train=True,eval=False):
        self.image_root = image_root
        self.train_paths = []
        self.test_paths = []
        self.train = train
        train_links = self.image_root+'/train.txt'
        test_links = self.image_root + '/test.txt'
        if eval:
            test_links = self.image_root+'/valid.txt'
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

class PairedDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,sketch_root,photo_root,transform=None,train=True,eval=False, fine_grain=False, balanced = False):
        'Initialization'
        self.balanced = balanced
        self.sketch_root = sketch_root
        self.photo_root = photo_root
        self.sketch_data, self.photo_data = [], []
        self.test_sketch,self.test_photo = [],[]
        self.label_to_indx_sketch = {}
        self.label_to_indx_photo = {}
        self.train = train
        self.fine_grain = fine_grain
        self._make_dataset()
        self.transform = transform

    def pil_loader(self,path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _make_dataset(self):

        self.classes = []
        self.class_to_index = {}
        self.length = 0
        if self.train:
            links= self.sketch_root+'/train.txt'
        else:
            links = self.sketch_root+'/test.txt'
        class_i =0
        with open(links) as f:
            for i,line in enumerate(f):
                label,path = line.split()

                if label not in self.class_to_index:
                    self.class_to_index[label] = class_i
                    self.classes.append(label)
                    class_i+=1
                label = self.class_to_index[label]
                self.sketch_data.append((label, path))

                if label not in self.label_to_indx_sketch:
                    self.label_to_indx_sketch[label]=[]

                self.label_to_indx_sketch[label].append(i)

        self.length = len(self.sketch_data)

        if self.train:
            links = self.photo_root + '/train.txt'
        else:
            links = self.photo_root + '/test.txt'

        with open(links) as f:
            for i,line in enumerate(f):
                label, path = line.split()
                label = self.class_to_index[label]
                if label not in self.label_to_indx_photo:
                    self.label_to_indx_photo[label] = []
                self.label_to_indx_photo[label].append(i)
                self.photo_data.append((label, path))
        self.classes_set = set(self.classes)

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if not self.train:
            np.random.seed(682)
        else:
            np.random.seed()
        target = np.random.randint(0, 2)
        if self.balanced:
            uni_prob = 1/len(self.classes)
            target = np.random.choice([0,1],p=[uni_prob,1-uni_prob])
        label_s,sketch = self.sketch_data[index]


        if target ==1:
            photo_index = np.random.choice(self.label_to_indx_photo[label_s])
            label_p,photo = self.photo_data[photo_index]
        else:
            neg_class = np.random.choice(list(self.classes_set-set([self.classes[label_s]])))
            neg_class = self.class_to_index[neg_class]
            photo_index = np.random.choice(self.label_to_indx_photo[neg_class])
            label_p,photo = self.photo_data[photo_index]


        sketch =self.pil_loader(sketch)
        photo = self.pil_loader(photo)
        if self.transform is not None:
            sketch=self.transform(sketch)
            photo = self.transform(photo)
        # return (sketch,photo,label_s,label_p), (label, target)

        return (sketch, photo), (target, label_s, label_p )