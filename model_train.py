import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import dataloader


class Net(nn.Module):
    # def __init__(self):  # Constructor
    #
    #     super(Net, self).__init__()
    #
    #     # 2 Layers
    #     self.fc1 = nn.Linear(3072, 800)
    #     self.fc2 = nn.Linear(800, 100)
    #     self.fc3 = nn.Linear(100, 10)
    #
    #     # xavier_initialization
    #
    # def forward(self, x):
    #     # x - shape = [64, 784]
    #     # Layer 1
    #     l1 = self.fc1(x)  # l1 - shape = [64, 100]
    #
    #     # Activation 1
    #     l1_a1 = torch.relu(l1)  # l1_a1 - shape = [64, 100]
    #
    #     l2 = self.fc2(l1_a1)
    #
    #     l2_a2 = torch.relu(l2)
    #
    #     # Layer 2
    #     l3 = self.fc3(l2_a2)  # l2 - shape = [64, 10]
    #
    #     # Activation 2
    #     l3_a3 = torch.relu(l3) # l2_a2 - shape = [64, 10]
    #
    #     return l3_a3

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        # self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(32*32, 32*32)
        self.fc1 = nn.Linear(32*32, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 10)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 32)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        # x = self.dropout1(x)
        # x = self.fc4(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def train(model, use_cuda, train_loader, optimizer, epoch):
    model.train()  # Tell the model to prepare for training
    print('training')

    for batch_idx, (data, target) in enumerate(train_loader):  # Get the batch

        # print(data.shape)
        # exit(0)
        # Converting the target to one-hot-encoding from categorical encoding
        # Converting the data to [batch_size, 784] from [batch_size, 1, 28, 28]
        # print('start train')

        # y_onehot = torch.zeros([target.shape[0], 10])  # Zero vector of shape [64, 10]
        # y_onehot[range(target.shape[0]), target] = 1

        # data = data.view([data.shape[0], 3072])
        # print(data.shape);
        # exit(0)
        # print(batch_idx)

        if use_cuda:
            data, target = data.cuda(), target.cuda()  # Sending the data to the GPU
            # data, y_onehot = data.cuda(), y_onehot.cuda()  # Sending the data to the GPU

        optimizer.zero_grad()  # Setting the cumulative gradients to 0
        output = model(data)  # Forward pass through the model
        # loss = torch.mean((output - y_onehot) ** 2)  # Calculating the loss
        loss = F.cross_entropy(output, target)
        loss.backward()  # Calculating the gradients of the model. Note that the model has not yet been updated.
        optimizer.step()  # Updating the model parameters. Note that this does not remove the stored gradients!

        # batch_idx = train_loader.batch_size
        if (batch_idx+1) % 2 == 0 or batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, use_cuda, test_loader):
    model.eval()  # Tell the model to prepare for testing or evaluation

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Tell the model that gradients need not be calculated
        for data, target in test_loader:  # Get the batch

            # Converting the target to one-hot-encoding from categorical encoding
            # Converting the data to [batch_size, 784] from [batch_size, 1, 28, 28]

            # y_onehot = torch.zeros([target.shape[0], 10])
            # y_onehot[range(target.shape[0]), target] = 1
            # data = data.view([data.shape[0], 3072])

            if use_cuda:
                data, target = data.cuda(), target.cuda()  # Sending the data to the GPU
                # data, target, y_onehot = data.cuda(), target.cuda(), y_onehot.cuda()  # Sending the data to the GPU

            # argmax([0.1, 0.2, 0.9, 0.4]) => 2
            # output - shape = [1000, 10], argmax(dim=1) => [1000]
            output = model(data)  # Forward pass
            # test_loss += torch.sum((output - y_onehot) ** 2)  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the maximum output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Get total number of correct samples

    test_loss /= len(test_loader.dataset)  # Accuracy = Total Correct / Total Samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def seed(seed_value):
    # This removes randomness, makes everything deterministic

    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    use_cuda = False  # Set it to False if you are using a CPU

    seed(0)  # Used to fix randomness in the code! Very important!

    # Convert the dataset to tensor and subtract the mean and divide by standard
    # deviation. Why? So that neurons don't saturate!
    # transform_train = transforms.Compose(
    #     [transforms.Resize((32, 32)),  # resizes the image so it can be perfect for our model.
    #      transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
    #      transforms.RandomRotation(10),  # Rotates the image to a specified angel
    #      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
    #      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
    #      transforms.ToTensor(),  # convert the image to tensor so that it can work with torch
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
    #      ])

    # transform = transforms.Compose([transforms.Resize((32, 32)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                                 ])
    # [a1, a2, a3] / [b1, b2, b3] = [a1/b1, a2/b2, a3/b3]

    # Wx + b =~ b

    # dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)  # Get the train dataset
    # dataset2 = datasets.MNIST('../data', train=False, transform=transform)  # Get the test dataset

    # training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  #
    # Data augmentation is only done on training images test_dataset = datasets.CIFAR10(root='./data', train=False,
    # download=True, transform=transform)

    train_dataset = dataloader.getDataLoader()
    test_dataset = dataloader.getDataLoader('test')

    # exit(0)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    # # Batch size of 100 i.e to work with 100 images at a time
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = Net()  # Get the model

    if use_cuda:
        model = model.cuda()  # Put the model weights on GPU

    # criterion = nn.CrossEntropyLoss()
    # same as categorical_crossentropy loss used in Keras models which runs on Tensorflow
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # fine tuned the lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.9)

    for epoch in range(1, config.numEpochs + 1):
        print(f'epoch={epoch}')
        train(model, use_cuda, train_dataset, optimizer, epoch)  # Train the network
        test(model, use_cuda, test_dataset)  # Test the network

    saveModel(model)

    # torch.save(model.state_dict(), "cifar.pt")
    # model.load_state_dict(torch.load('cifar.pt'))
    # Loading a saved model - model.load_state_dict(torch.load('mnist_cnn.pt'))


def saveModel(model: Net):
    torch.save(model.state_dict(), f'{config.savePath}/model_final_epochs_{config.numEpochs}.pt')
    # save model in onnx format
    inp = torch.randn(1, 1, 32, 32)
    torch.onnx.export(model, inp, f'{config.savePath}/model_final_user1.onnx', verbose=True,
                      input_names=['data'], output_names=['output'])


if __name__ == '__main__':
    main()
