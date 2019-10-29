import torch
import torch.nn as nn
import torch.nn.functional as F

def hswish(x) :
    x = x * (F.relu6(x + 3) / 6)
    return x

class bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, expand_channel, kernel_size, stride, hswish=False, se=False):
        super(bottleneck, self).__init__()

        self.se = se

        self.activateFn = F.relu
        if hswish:
            self.activateFn = hswish

        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

        self.conv1 = nn.Conv2d(in_channel, expand_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channel)

        self.conv2 = nn.Conv2d(expand_channel, expand_channel, kernel_size=kernel_size, stride=stride, padding=0, groups=expand_channel, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channel)

        self.conv3 = nn.Conv2d(expand_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if self.se:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(expand_channel, expand_channel//4)
            self.fc2 = nn.Linear(expand_channel//4, expand_channel)
            self.hsigmoid = nn.hardtanh(0, 1)

    def forward(self, x):
        shortcut = self.bn(self.shortcut(x))

        x = self.activateFn(self.bn1(self.conv1(x)))
        x = self.activateFn(self.bn2(self.conv2(x)))

        if self.se:
            x_se = self.pooling(x)
            x_se = self.fc1(x_se)
            x_se = F.relu(x_se)
            x_se = self.fc2(x_se)
            x_se = self.hsigmoid(x_se)

            x = torch.matmul(x, x_se)

        x = self.activateFn(self.bn3(self.conv3(x)))

        x = x + shortcut

        return x

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.bneck1 = bottleneck(in_channel=16, out_channel=16, expand_channel=16, kernel_size=3, stride=2, hswish=False, se=True)
        self.bneck2 = bottleneck(in_channel=16, out_channel=24, expand_channel=72, kernel_size=3, stride=2, hswish=False, se=False)
        self.bneck3 = bottleneck(in_channel=24, out_channel=24, expand_channel=88, kernel_size=3, stride=1, hswish=False, se=False)
        self.bneck4 = bottleneck(in_channel=24, out_channel=40, expand_channel=96, kernel_size=5, stride=2, hswish=True, se=True)
        self.bneck5 = bottleneck(in_channel=40, out_channel=40, expand_channel=240, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck6 = bottleneck(in_channel=40, out_channel=40, expand_channel=240, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck7 = bottleneck(in_channel=40, out_channel=48, expand_channel=120, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck8 = bottleneck(in_channel=48, out_channel=48, expand_channel=144, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck9 = bottleneck(in_channel=48, out_channel=96, expand_channel=288, kernel_size=5, stride=2, hswish=True, se=True)
        self.bneck10 = bottleneck(in_channel=96, out_channel=96, expand_channel=576, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck11 = bottleneck(in_channel=96, out_channel=96, expand_channel=576, kernel_size=5, stride=1, hswish=True, se=True)

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)

        self.pooling_se = nn.AdaptiveAvgPool2d(1)
        self.fc1_se = nn.Linear(576, 576//4)
        self.fc2_se = nn.Linear(576//4, 576)
        self.hsigmoid_se = nn.hardtanh(0, 1)

        self.pooling = nn.AvgPool2d(7)

        self.conv3 = nn.Conv2d(576, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = hswish(self.bn1(self.conv1(x)))
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)
  
        x = hswish(self.bn2(self.conv2(x)))
        x_se = self.pooling_se(x)
        x_se = self.fc1_se(x)
        x_se = F.relu(x_se)
        x_se = self.fc2_se(x)
        x_se = self.hsigmoid_se(x)

        x = torch.matmul(x, x_se)
        
        x = self.pooling(x)
        x = hswish(self.conv3(x))
        x = self.conv3(x)

        return x

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.bneck1 = bottleneck(in_channel=16, out_channel=16, expand_channel=16, kernel_size=3, stride=1, hswish=False, se=False)
        self.bneck2 = bottleneck(in_channel=16, out_channel=24, expand_channel=64, kernel_size=3, stride=2, hswish=False, se=False)
        self.bneck3 = bottleneck(in_channel=24, out_channel=24, expand_channel=72, kernel_size=3, stride=1, hswish=False, se=False)
        self.bneck4 = bottleneck(in_channel=24, out_channel=40, expand_channel=72, kernel_size=5, stride=2, hswish=False, se=True)
        self.bneck5 = bottleneck(in_channel=40, out_channel=40, expand_channel=120, kernel_size=5, stride=1, hswish=False, se=True)
        self.bneck6 = bottleneck(in_channel=40, out_channel=40, expand_channel=120, kernel_size=5, stride=1, hswish=False, se=True)
        self.bneck7 = bottleneck(in_channel=40, out_channel=80, expand_channel=240, kernel_size=3, stride=2, hswish=True, se=False)
        self.bneck8 = bottleneck(in_channel=80, out_channel=80, expand_channel=200, kernel_size=3, stride=1, hswish=True, se=False)
        self.bneck9 = bottleneck(in_channel=80, out_channel=80, expand_channel=184, kernel_size=3, stride=1, hswish=True, se=False)
        self.bneck10 = bottleneck(in_channel=80, out_channel=80, expand_channel=184, kernel_size=3, stride=1, hswish=True, se=False)
        self.bneck11 = bottleneck(in_channel=80, out_channel=112, expand_channel=480, kernel_size=3, stride=1, hswish=True, se=True)
        self.bneck12 = bottleneck(in_channel=112, out_channel=112, expand_channel=672, kernel_size=3, stride=1, hswish=True, se=True)
        self.bneck13 = bottleneck(in_channel=112, out_channel=160, expand_channel=672, kernel_size=5, stride=2, hswish=True, se=True)
        self.bneck14 = bottleneck(in_channel=160, out_channel=160, expand_channel=960, kernel_size=5, stride=1, hswish=True, se=True)
        self.bneck15 = bottleneck(in_channel=160, out_channel=160, expand_channel=960, kernel_size=5, stride=1, hswish=True, se=True)

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.pooling = nn.AvgPool2d(7)

        self.conv3 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = hswish(self.bn1(self.conv1(x)))
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)
        x = self.bneck12(x)
        x = self.bneck13(x)
        x = self.bneck14(x)
        x = self.bneck15(x)
        x = hswish(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        x = hswish(self.conv3(x))
        x = self.conv3(x)

        return x
