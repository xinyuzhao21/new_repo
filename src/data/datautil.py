import torch
import os
import random
import matplotlib.pyplot as plt

def train_test_split(dir: str, split=(0.9, 0.1)):
    seed = 682
    random.seed(seed)
    train,total_train = [],0
    test,total_test= [],0
    labels = []
    # https://stackoverflow.com/questions/17412439/how-to-split-data-into-trainset-and-testset-randomly
    for class_dir in os.scandir(dir):
        if  not os.path.isdir(class_dir):
            continue
        label = class_dir.name
        labels.append(str(label))
        samples = [image_path.path for image_path in os.scandir(class_dir.path)]
        random.shuffle(samples)
        num_train, num_test = map(lambda x: int(x * len(samples)), split)
        train_sample = samples[:num_train]
        test_sample = samples[num_train:]
        total_train+=len(train_sample)
        total_test += len(test_sample)
        train += [(label, train_sample)]
        test += [(label, test_sample)]
    print("train",total_train,"test",total_test, "total", total_train+total_test)
    with open(dir+'/train.txt', 'w') as f:
        for label, path in train:
            for p in path:
                f.write(label + " " + p + '\n')
    with open(dir+'/test.txt', 'w') as f:
        for label, path in test:
            for p in path:
                f.write(label + " " + p + '\n')
    with open('labels.txt', 'w') as f:
        f.writelines('\n'.join(labels))

def showImage(image):
    plt.imshow(image)
    plt.show()