from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config


class dataloader(Dataset):

    def __init__(self):

        self.allClasses = sorted(os.listdir(config.basePath))

        self.allData = {}

        for imgClass in self.allClasses:
            self.allData[imgClass] = []
            imgPath = sorted(os.listdir(config.basePath + '/' + imgClass + '/image/'))
            for image in imgPath:
                image = self.process(config.basePath + '/' + imgClass + '/image/' + image)
                self.allData[imgClass].append(image)

    def process(self, path):

        image = Image.fromarray(plt.imread(path)[:, :, 3])
        image = image.resize((128, 128))
        image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

        return image

    def __getitem__(self, item):

        classID = np.random.randint(len(self.allData))
        className = self.allClasses[classID]
        image_id = np.random.randint(len(self.allData[className]))
        image = self.allData[className][image_id]

        return image, classID

    def __len__(self):
        return config.iterations[self.type]

    def _worker_init_fn(worker_id):
        np.random.seed(worker_id)

    def getDataLoader(type_='train'):

        return DataLoader(
            dataloader(),
            batch_size=config.batchSize[type_],
            num_workers=config.numWorkers[type_],
            worker_init_fn=_worker_init_fn
        )
