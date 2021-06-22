'''
fpn+attention network
'''
import torch
import torch.nn as nn

class ResnetAttention(nn.Module):
    def __init__(self, model, num_classes):
        super(ResnetAttention, self).__init__()

        self.L = 256
        self.D = 128
        self.K = num_classes

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.lateral_conv_1 = nn.Conv2d(64, 512, 1)
        self.lateral_conv_2 = nn.Conv2d(128, 512, 1)
        self.lateral_conv_3 = nn.Conv2d(256, 512, 1)
        self.lateral_conv_4 = nn.Conv2d(512, 512, 1)

        self.attention_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.se_fc_1 = nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.attention_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.se_fc_2 = nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.attention_pool_3 = nn.AdaptiveAvgPool2d(1)
        self.se_fc_3 = nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.attention_pool_4 = nn.AdaptiveAvgPool2d(1)
        self.se_fc_4 = nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        for n in [self.se_fc_1, self.se_fc_2, self.se_fc_3, self.se_fc_4]:
            for m in n:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

        self.upsample_1 = nn.AdaptiveAvgPool2d((7, 7))
        self.upsample_2 = nn.AdaptiveAvgPool2d((7, 7))
        self.upsample_3 = nn.AdaptiveAvgPool2d((7, 7))
        self.upsample_4 = nn.AdaptiveAvgPool2d((7, 7))

        self.conv1_1 = nn.Conv2d(512, 512, 1)
        self.conv1_2 = nn.Conv2d(512, 512, 1)
        self.conv1_3 = nn.Conv2d(512, 512, 1)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1_1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, self.L),
            nn.ReLU(),
        )
        self.fc1_2 = nn.Sequential(
            nn.Linear(512 * 1 * 1, self.L),
            nn.ReLU(),
        )
        self.fc1_3 = nn.Sequential(
            nn.Linear(512 * 1 * 1, self.L),
            nn.ReLU(),
        )

        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        self.attention3 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

        self.fc2_1 = nn.Sequential(
            nn.Linear(self.L, self.K),
        )
        self.fc2_2 = nn.Sequential(
            nn.Linear(self.L, self.K),
        )
        self.fc2_3 = nn.Sequential(
            nn.Linear(self.L, self.K),
        )


        self.w = nn.Parameter(torch.tensor([[1., 1., 1.]]), requires_grad=True)
    def forward(self, x):
        x = x.squeeze(0) #(15,3,224,224)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) # (15,64,56,56)

        x1 = self.lateral_conv_1(x)
        b, c, h, w = x1.size() # b=15 c=512
        w1 = self.attention_pool_1(x1).view(b, c)
        w1 = self.se_fc_1(w1).view(b, 1) # 15,1
        w1 = w1.unsqueeze(dim=1).unsqueeze(dim=2).expand(b, c, 7, 7)
        x1 = self.upsample_1(x1)
        x1 = x1 + x1 * w1

        x = self.layer2(x) # (15,128,28,28)

        x2 = self.lateral_conv_2(x)
        b, c, h, w = x2.size()
        w2 = self.attention_pool_2(x2).view(b, c)
        w2 = self.se_fc_2(w2).view(b, 1)
        w2 = w2.unsqueeze(dim=1).unsqueeze(dim=2).expand(b, c, 7, 7)
        x2 = self.upsample_2(x2)
        x2 = x2 + x2 * w2

        x = self.layer3(x) # (15,256,14,14)

        x3 = self.lateral_conv_3(x)
        b, c, h, w = x3.size()
        w3 = self.attention_pool_3(x3).view(b, c)
        w3 = self.se_fc_3(w3).view(b, 1)
        w3 = w3.unsqueeze(dim=1).unsqueeze(dim=2).expand(b, c, 7, 7)
        x3 = self.upsample_3(x3)
        x3 = x3 + x3 * w3

        x = self.layer4(x) # (15,512,7,7)

        x4 = self.lateral_conv_4(x)
        b, c, h, w = x4.size()
        w4 = self.attention_pool_4(x4).view(b, c)
        w4 = self.se_fc_4(w4).view(b, 1)
        w4 = w4.unsqueeze(dim=1).unsqueeze(dim=2).expand(b, c, 7, 7)
  
        x4 = x4 + x4 * w4
        # x4 = self.upsample_1(x4)

        x4 = torch.add(x4, x3)
        x4 = torch.add(x4, x2)
        x4 = torch.add(x4, x1)  # (15,512,7,7)

        # 7*7 branch
        x = self.conv1_1(x4)
        x = self.avgpool1(x)

        x = x.view(-1, 512 * 1 * 1)
        x = self.fc1_1(x) #(15,256)

        a = self.attention1(x) # (15,1)
        a = torch.transpose(a, 1, 0)
        softmax = nn.Softmax(dim=1)
        a = softmax(a)

        m = torch.mm(a, x)

        m = m.view(-1, 1 * 256)
        result = self.fc2_1(m)

        return result
