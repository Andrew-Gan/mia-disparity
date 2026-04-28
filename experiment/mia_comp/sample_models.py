import torch
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        layers = torch.nn.ModuleList()

        conv_layer = []
        conv_layer.append(torch.nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(torch.nn.BatchNorm2d(channels))
        conv_layer.append(torch.nn.ReLU(inplace=True))
        conv_layer.append(torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(torch.nn.BatchNorm2d(channels))

        layers.append(torch.nn.Sequential(*conv_layer))

        shortcut = torch.nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(torch.nn.ReLU(inplace=True))

        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x)
        fwd += self.layers[1](x)
        fwd = self.layers[2](fwd)
        return fwd


class ResNet(torch.torch.nn.Module):
    def __init__(self, num_blocks, num_classes, input_size, block_type='basic'):
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.input_size = input_size
        self.block_type = block_type
        self.in_channels = 16
        self.num_output = 1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        init_conv.append(torch.nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))

        init_conv.append(torch.nn.BatchNorm2d(self.in_channels))
        init_conv.append(torch.nn.ReLU(inplace=True))

        self.init_conv = torch.nn.Sequential(*init_conv)

        self.layers = torch.nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        end_layers.append(torch.nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(torch.nn.Linear(64 * self.block.expansion, self.num_classes))
        self.end_layers = torch.nn.Sequential(*end_layers)


    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers


    def forward(self, x, k=None, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        if k is None:
            out = self.init_conv(x)

            for layer in self.layers:
                out = layer(out)

            out = self.end_layers(out)

            return out

        # the following is for getting feature maps
        out = self.init_conv(x)

        n_layer = 0
        _fm = None

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if not train:
                if isinstance(layer, BasicBlock):
                    if n_layer == k:
                        return None, out.view(out.size(0), -1)
                    n_layer += 1

        out = self.end_layers(out)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(out, 1)
                return None, _fm.view(_fm.size(0), -1)
        else:
            return out


def create_resnet56(num_classes=10, input_size=32):
    num_blocks = [9, 9, 9]
    num_classes = num_classes
    input_size = input_size
    block_type = 'basic'
    return ResNet(num_blocks=num_blocks, num_classes=num_classes, input_size=input_size, block_type=block_type)


class wide_basic(torch.nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.layers = torch.nn.ModuleList()
        conv_layer = []
        conv_layer.append(torch.nn.BatchNorm2d(in_channels))
        conv_layer.append(torch.nn.ReLU(inplace=True))
        conv_layer.append(torch.nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
        conv_layer.append(torch.nn.Dropout(p=dropout_rate))
        conv_layer.append(torch.nn.BatchNorm2d(channels))
        conv_layer.append(torch.nn.ReLU(inplace=True))
        conv_layer.append(torch.nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True))

        self.layers.append(torch.nn.Sequential(*conv_layer))

        shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True),
            )

        self.layers.append(shortcut)

    def forward(self, x):
        out = self.layers[0](x)
        out += self.layers[1](x)
        return out


class WideResNet(torch.nn.Module):
    def __init__(self, num_blocks, widen_factor, num_classes, dropout_rate, input_size):
        super(WideResNet, self).__init__()
        self.num_blocks = num_blocks
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.in_channels = 16
        self.num_output = 1

        self.init_conv = torch.nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.layers = torch.nn.ModuleList()
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels * self.widen_factor, block_id=0, stride=1))
        self.layers.extend(self._wide_layer(wide_basic, 32 * self.widen_factor, block_id=1, stride=2))
        self.layers.extend(self._wide_layer(wide_basic, 64 * self.widen_factor, block_id=2, stride=2))

        end_layers = []

        end_layers.append(torch.nn.BatchNorm2d(64 * self.widen_factor, momentum=0.9))
        end_layers.append(torch.nn.ReLU(inplace=True))
        end_layers.append(torch.nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(torch.nn.Linear(64 * self.widen_factor, self.num_classes))
        self.end_layers = torch.nn.Sequential(*end_layers)

        self.initialize_weights()

    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.dropout_rate, stride))
            self.in_channels = channels
        return layers

    def forward(self, x, k=0, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        if k is None:
            out = self.init_conv(x)

            for layer in self.layers:
                out = layer(out)

            out = self.end_layers(out)

            return out

        # the following is for getting feature maps
        out = self.init_conv(x)

        n_layer = 0
        _fm = None

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if not train:
                if isinstance(layer, wide_basic):
                    if n_layer == k:
                        return None, out.view(out.size(0), -1)
                    n_layer += 1

        out = self.end_layers(out)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(out, 1)
                return None, _fm.view(_fm.size(0), -1)
        else:
            return out


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_features(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        return out

def create_wideresnet28_2(num_classes=10, input_size=32):
    num_blocks = [4, 4, 4]
    widen_factor = 2
    dropout_rate = 0.3
    return WideResNet(num_blocks=num_blocks, widen_factor=widen_factor, num_classes=num_classes,
                      dropout_rate=dropout_rate, input_size=input_size)