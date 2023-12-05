import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.Linears = nn.Sequential(
            nn.Linear(inputSize, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, outputSize)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.Linears(x)
    
class VGG16(nn.Module):
    def __init__(self, inChannels, numOfClass):
        super().__init__()
        self.ReLU = nn.ReLU()

        self.Convolutions = nn.Sequential(
            nn.Conv2d(in_channels = inChannels, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.Linear0 = nn.Linear(512, 4096)
        self.Dropout0 = nn.Dropout()
        self.Linear1 = nn.Linear(4096, 4096)
        self.Dropout1 = nn.Dropout()
        self.Linear2 = nn.Linear(4096, numOfClass)

    def forward(self, x):
        x = self.Convolutions(x)

        x = x.view(x.size(0), -1)
        
        x = self.Linear0(x)
        x = self.ReLU(x)
        x = self.Dropout0(x)

        x = self.Linear1(x)
        x = self.ReLU(x)
        x = self.Dropout1(x)

        x = self.Linear2(x)
        return x
