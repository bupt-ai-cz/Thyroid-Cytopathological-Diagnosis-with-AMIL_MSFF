'''
fpn network
'''
import torch
import torch.nn as nn
class ResnetAttention(nn.Module):
    def __init__(self, model, num_classes):
        super(ResnetAttention, self).__init__()

        self.L = 256
        self.D = 128
        self.K = num_classes

        # Resnet50
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Lateral connection
        self.lateral_conv_1 = nn.Conv2d(64, 512, 1, bias=False)
        self.lateral_conv_2 = nn.Conv2d(128, 512, 1, bias=False)
        self.lateral_conv_3 = nn.Conv2d(256, 512, 1, bias=False)
        self.lateral_conv_4 = nn.Conv2d(512, 512, 1, bias=False)

        nn.init.kaiming_normal_(self.lateral_conv_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.lateral_conv_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.lateral_conv_3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.lateral_conv_4.weight, mode='fan_out', nonlinearity='relu')

        self.upsample_1 = nn.AdaptiveAvgPool2d((7, 7))
        self.upsample_2 = nn.AdaptiveAvgPool2d((7, 7))
        self.upsample_3 = nn.AdaptiveAvgPool2d((7, 7))

        self.conv1_1 = nn.Conv2d(512, 512, 1, bias=True)
        nn.init.kaiming_normal_(self.conv1_1.weight, mode='fan_out', nonlinearity='relu')


        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1_1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, self.L),
            nn.ReLU(),
        )

        # nn.init.normal_(self.fc1_1[0].weight)

        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

        # nn.init.normal_(self.attention1[0].weight)
        # nn.init.normal_(self.attention1[2].weight)


        self.fc2_1 = nn.Sequential(
            nn.Linear(self.L, self.K),
        )

        # nn.init.normal_(self.fc2_1[0].weight)

    def forward(self, x):

        x = x.squeeze(0) #(15,3,224,224)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) # (15,64,56,56)

        x1 = self.lateral_conv_1(x)
        x1 = self.upsample_1(x1)

        x = self.layer2(x) # (15,128,28,28)

        x2 = self.lateral_conv_2(x)
        x2 = self.upsample_2(x2)

        x = self.layer3(x) # (15,256,14,14)

        x3 = self.lateral_conv_3(x)
        x3 = self.upsample_3(x3)

        x = self.layer4(x) # (15,512,7,7)

        x4 = self.lateral_conv_4(x)

        x4 = torch.add(x4, x3)
        x4 = torch.add(x4, x2)
        x4 = torch.add(x4, x1)  # (15,512,7,7)

        # main branch
        x = self.conv1_1(x4)
        x = self.avgpool1(x)

        x = x.view(-1, 512 * 1 * 1)
        x = self.fc1_1(x) #(15,256)

        a = self.attention1(x)
        a = torch.transpose(a, 1, 0)
        softmax = nn.Softmax(dim=1)
        a = softmax(a)

        m = torch.mm(a, x)

        m = m.view(-1, 1 * 256)
        result = self.fc2_1(m)

        return result
