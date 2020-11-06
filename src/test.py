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
tmp_root ='../test'
util.train_test_split(tmp_root,split=(0.8,0.1,0.1))
train_data = DataSet.ImageDataset(tmp_root,transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, shuffle=True)

for x_batch, y_batch in train_loader:
    print(x_batch.shape)
    to_image = transforms.ToPILImage()
    image =to_image(x_batch[0])
    util.showImage(image)
    print(train_data.class_to_idx)
    print(y_batch)