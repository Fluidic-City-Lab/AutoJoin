import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureXNvidia(nn.Module):
    def __init__(self, in_channel=3):
        super(FeatureXNvidia, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 24, 5, 2)
        self.clean_bn1 = nn.BatchNorm2d(24)
        self.adv_bn1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.clean_bn2 = nn.BatchNorm2d(36)
        self.adv_bn2 = nn.BatchNorm2d(36)

        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.clean_bn3 = nn.BatchNorm2d(48)
        self.adv_bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 64, 3)
        self.clean_bn4 = nn.BatchNorm2d(64)
        self.adv_bn4 = nn.BatchNorm2d(64)
    
    def forward(self, x, tag="clean"):
        x = x/127.5 - 1.0

        out = self.conv1(x)
        if tag == "clean":
            out = self.clean_bn1(out)
        else:
            out = self.adv_bn1(out)
        out = F.elu(out)

        out = self.conv2(out)
        if tag == "clean":
            out = self.clean_bn2(out)
        else:
            out = self.adv_bn2(out)
        out = F.elu(out)

        out = self.conv3(out)
        if tag == "clean":
            out = self.clean_bn3(out)
        else:
            out = self.adv_bn3(out)
        out = F.elu(out)

        out = self.conv4(out)
        if tag == "clean":
            out = self.clean_bn4(out)
        else:
            out = self.adv_bn4(out)
        out = F.elu(out)

        return out

class HeadNvidia(nn.Module):
    def __init__(self):
        super(HeadNvidia, self).__init__()

        self.conv5 = nn.Conv2d(64, 64, 3)
        self.clean_bn5 = nn.BatchNorm2d(64)
        self.adv_bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
    
    def forward(self, x, tag="clean"):
        
        out = self.conv5(x)
        if tag == "clean":
            out = self.clean_bn5(out)
        else:
            out = self.adv_bn5(out)
        out = F.elu(out)

        out = out.reshape(-1, 64 * 1 * 18)
        out = F.elu(self.fc1(out))
        out = F.elu(self.fc2(out))
        out = F.elu(self.fc3(out))
        out = self.fc4(out)

        return out


class NvidiaAdvBN(nn.Module):
    def __init__(self, nChannel=3):
        super(NvidiaAdvBN, self).__init__()
        
        self.feature_x = FeatureXNvidia()
        self.head = HeadNvidia()

    def _forward_impl(self, x, stage='full', tag='clean'):
        if stage == 'feature_x':
            x = self.feature_x(x)
        elif stage == 'head':
            x = self.head(x, tag)
        else:
            x = self.feature_x(x)
            x = self.head(x, tag)
        return x

    def forward(self, x, stage='full', tag='clean'):
        return self._forward_impl(x, stage, tag)