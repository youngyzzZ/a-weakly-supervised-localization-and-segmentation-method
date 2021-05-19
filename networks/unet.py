import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from networks.ops import initialize_weights
from torch.nn import DataParallel


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        # if self.pooling:
        #     self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if self.pooling:
            self.pool = conv3x3(self.out_channels, self.out_channels, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x

        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        super(UNet, self).__init__()

        assert up_mode in ('transpose', 'upsample')
        self.up_mode = up_mode

        assert merge_mode in ('concat', 'add')
        self.merge_mode = merge_mode

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        assert not (self.up_mode == 'upsample' and self.merge_mode == 'add')

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []
        self.up_convs_1 = []

        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)
            # if i >= 2:
            self.up_convs_1.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        self.conv_final_1 = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.up_convs_1 = nn.ModuleList(self.up_convs_1)

        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

        # initialize_weights(self)

    def forward(self, x):
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        x1 = x.clone()
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            # if i == 2:
            #     model_1 = self.up_convs_1[i - 2]
            #     x1 = model_1(before_pool, x)
            # elif i > 2:
            #     model_1 = self.up_convs_1[i - 2]
            #     x1 = model_1(before_pool, x1)
            module_1 = self.up_convs_1[i]
            x = module(before_pool, x)
            x1 = module_1(before_pool, x1)

        x = self.conv_final(x)
        x1 = self.conv_final_1(x1)
        return self.tanh(x), self.tanh(x1)
        # return self.sigmoid(x), self.sigmoid(x1)


if __name__ == "__main__":
    model = UNet(1, depth=5, in_channels=1)
    print(model)
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    if torch.cuda.is_available():
        x = torch.randn((8, 1, 128, 128)).cuda()
    else:
        x = torch.randn((8, 1, 128, 128))
    out = model(x)
    # print(out)
    print(out[0].shape)
