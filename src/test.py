import src.data.dataset as DataSet
import src.data.datautil as util
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
util.train_test_split(tmp_root,split=(0.75,0.25))
train_data = DataSet.ImageDataset(tmp_root,transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, shuffle=True)
to_image = transforms.ToPILImage()
# for x_batch, y_batch in train_loader:
#     print(x_batch.shape)
#     image =to_image(x_batch[0])
#     util.showImage(image)
#     print(y_batch)