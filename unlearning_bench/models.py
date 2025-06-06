# external deps
import torch
import torchvision


class Mul(torch.nn.Module):

    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):

    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True),
    )


def ResNet9(num_classes=10, channels_last=True):
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def ResNet18(num_classes=10,
             pretrained=False,
             channels_last=True,
             ckpt_path=None):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes, bias=True)

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state, strict=True)

    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


MODELS = {
    "resnet9": ResNet9,
    "resnet18": ResNet18,
}

if __name__ == "__main__":
    pass
