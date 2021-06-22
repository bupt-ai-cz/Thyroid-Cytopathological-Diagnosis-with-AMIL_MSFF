'''
baseline network
'''
import torch
import torch.nn as nn

class ResnetAttention(nn.Module):
    def __init__(self, model, num_classes):
        super(ResnetAttention, self).__init__()

        self.L = 256
        self.D = 128
        self.K = num_classes

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.L, self.K),
        )

    def forward(self, x):
        x = x.squeeze(0) #(15,3,224,224)
        x = self.features(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc1(x) #(15,256)

        a = self.attention(x)
        a = torch.transpose(a, 1, 0)
        softmax = nn.Softmax(dim=1)
        a = softmax(a)

        m = torch.mm(a, x)

        m = m.view(-1, 1 * 256)
        result = self.fc2(m)

        return result