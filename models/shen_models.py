import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class net_resnet_pytorch(nn.Module):
    def __init__(self, nChannel=3):
        super(net_resnet_pytorch, self).__init__()
        # self.conv1 = nn.Conv2d(nChannel, 3, 1)
        self.resnet50 = models.resnet50()
        self.resnet50.fc = nn.Linear(2048,1)
        # self.header = nn.Linear(1000, 1)


    def forward(self, x):
        x = x / 255.0
        output = self.resnet50(x)

        return output, output

class net_nvidia_pytorch(nn.Module):
    def __init__(self, nChannel=3):
        super(net_nvidia_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(nChannel, 24, 5, 2)
        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
        # x = x / 255.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        #print(x.shape)
        x = x.reshape(-1, 64 * 1 * 18)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        last_layer_feature = F.elu(self.fc3(x))
        output = self.fc4(last_layer_feature)
        return output, last_layer_feature