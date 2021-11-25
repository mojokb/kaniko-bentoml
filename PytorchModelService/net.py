import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.float()
        h1 = f.relu(self.fc1(x.view(-1, 784)))
        h2 = self.fc2(h1)
        return h2 #f.softmax(h2, dim=1)
