import os

basePath = 'data'
lr = 1e-3

savePath = 'savedModel'

os.makedirs(savePath, exist_ok=True)

seed = 0
numEpochs = 50

batchSize = {
    'train': 10,
    'test': 20
}

numWorkers = {
    'train': 2,
    'test': 2
}

iterations = {
    'train': 1000,
    'test': 100
}
