import torch
import torch.nn as nn
import torch.nn.functional as F


class FCMNISTModel(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=128, seed=None):
        super(FCMNISTModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.W1(input)
        x_ = torch.tanh(x)
        logits = self.W2(x_)
        return logits

class FCCIFARModel(nn.Module):
    def __init__(self, input_dim=3072, output_dim=10, hidden_dim=128, seed=None):
        super(FCCIFARModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.W1(input)
        x = torch.tanh(x)
        x = self.W2(x)
        x = torch.tanh(x)
        logits = self.W3(x)
        return logits

class MNISTConv(nn.Module): # from https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, seed=None):
        super(MNISTConv, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Conv(nn.Module): # from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html (approx 63k params)
    def __init__(self, seed=None):
        super(CIFAR10Conv, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.l1 = 6
        self.l2 = 16
        self.h1 = 120
        self.h2 = 84
        self.conv1 = nn.Conv2d(3, self.l1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.l1, self.l2, 5)
        self.fc1 = nn.Linear(self.l2 * 5 * 5, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, 10)
        self.sp = nn.Softplus()
        # print("Num parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, x):
        # IPython.embed()
        x = self.sp(self.conv1(x))
        x = self.pool(x)
        x = self.sp(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.l2 * 5 * 5)
        x = self.sp(self.fc1(x))
        x = self.sp(self.fc2(x))
        x = self.fc3(x)
        return x


class MNISTAutoencoder(nn.Module):  # MNIST-A

    def __init__(self, seed=None):
        super(MNISTAutoencoder, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 1000, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(1000, 500, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(500, 250, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(250, 30, bias=True))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(30, 250, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(250, 500, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(500, 1000, bias=True),
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(1000, 28 * 28, bias=True), )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded