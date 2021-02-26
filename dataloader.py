from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image

import config


class dataloader(Dataset):

    splitType = ''

    def __init__(self):

        self.allClasses = sorted(os.listdir(config.basePath))

        self.allData = {}

        for imgClass in self.allClasses:
            self.allData[imgClass] = []
            imgPath = sorted(os.listdir(config.basePath + '/' + imgClass + '/image/'))
            for image in imgPath:
                image = self.process(config.basePath + '/' + imgClass + '/image/' + image, self.splitType)
                self.allData[imgClass].append(image)

    def process(self, path, split='train'):

        # image = Image.fromarray(plt.imread(path)[:, :, 3])
        # image = Image.open(path).convert('RGB')
        # print(image.size)
        image = Image.open(path)
        arr = np.asarray(image)
        image = Image.fromarray(arr[:, :, 3])
        # image = image.resize((128, 128))
        # print(type(image))
        # image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]
        # print(type(image))
        # image.show()

        # transform split
        if split == 'train':
            transform_train = transforms.Compose(
                [transforms.Resize((32, 32)),  # resizes the image so it can be perfect for our model.
                 transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                 transforms.RandomRotation(30),  # Rotates the image to a specified angel
                 transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                 # # Performs actions like zooms, change shear angles.
                 # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                 transforms.ToTensor(),  # convert the image to tensor so that it can work with torch
                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
                 transforms.Normalize((0.5,), (0.5,))
                 ])
            image = transform_train(image)
            # print(type(image))
        elif split == 'test':
            transforms_test = transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))
                 ])
            image = transforms_test(image)

        return image

    def __getitem__(self, item):

        classID = np.random.randint(len(self.allData))
        className = self.allClasses[classID]
        image_id = np.random.randint(len(self.allData[className]))
        image = self.allData[className][image_id]

        return image, classID

    def __len__(self):
        return config.iterations[self.splitType]


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


def getDataLoader(type_='train'):

    dataloader.splitType = type_

    return DataLoader(
        dataloader(),
        batch_size=config.batchSize[type_],
        num_workers=config.numWorkers[type_],
        worker_init_fn=_worker_init_fn
    )
