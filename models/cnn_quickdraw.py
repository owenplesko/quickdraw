import torch.nn as nn
import torch.nn.functional as F

class CNNQuickDraw(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNQuickDraw, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Same convolutions -> output -> same as input dimension
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # last conv out was 8 so this conv input is 8.

        self.fc1 = nn.Linear(16*7*7, num_classes)
        # in fc1 -> 16 bcoz that is outchannel in conv2 and 7*7 because -> 2 poolings will make it (28/2)/2 => 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x