"""
In this code,the difference between these classes is multiples of downsampling.
While-loop is strongly recommend.
reference: https://github.com/jaxony/unet-pytorch/blob/master/model.py
"""
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from networks.ops import BatchNorm, initialize_weights, avg_pooling


class MiddleMultiScale(nn.Module):
    """
    ablation experiment setting: remove the top feature output (last Conv2d + first Conv2d without downsampling)
    use feature map form middle layer rather than top layer
    """

    def __init__(self, depth, downsampling, global_pooling=avg_pooling):
        super(MiddleMultiScale, self).__init__()
        self.start_filts = 64
        self.input_dim = 3
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling
        self.global_pooling = global_pooling

        self.final_outs = 0
        down_conv = []
        down_conv.append(BatchNorm(self.input_dim, self.start_filts))
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)

        ins = 64
        for i in range(1, self.downsampling):
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
            ins = self.outs
        self.final_outs += self.outs
        self.down_convs = nn.Sequential(*down_conv)

        convs = []
        for _ in range(self.depth - self.downsampling):
            convs.append(BatchNorm(self.outs, self.outs))
        self.final_outs += self.outs
        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            nn.Linear(self.final_outs, self.output_dim)
        )
        initialize_weights(self)

    def forward(self, x):
        for idx, module in enumerate(self.down_convs, 1):
            x = module(x)
            if idx == self.downsampling:
                feature2 = x
        for module in self.convs:
            x = module(x)
        feature3 = x
        x = torch.cat((self.global_pooling(feature2), self.global_pooling(feature3)), 1)
        x = self.fc(x)
        return x


class TopMultiScale(nn.Module):
    def __init__(self, depth, downsampling, global_pooling=avg_pooling):
        """
        use the first downsampling block and the second downsampling block.
        use feature map from top layer rather middle layer
        :param depth:
        :param global_pooling:
        """

        super(TopMultiScale, self).__init__()
        self.start_filts = 64
        self.input_dim = 3
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling
        self.global_pooling = global_pooling

        self.final_outs = self.start_filts
        down_conv = []
        down_conv.append(BatchNorm(self.input_dim, self.start_filts))
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)

        ins = 64
        for i in range(1, self.downsampling):
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
            ins = self.outs
        self.down_convs = nn.Sequential(*down_conv)

        convs = []
        for _ in range(self.depth - self.downsampling):
            convs.append(BatchNorm(self.outs, self.outs))
        self.final_outs += self.outs
        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            nn.Linear(self.final_outs, self.output_dim)
        )
        initialize_weights(self)

    def forward(self, x):
        for idx, module in enumerate(self.down_convs, 1):
            x = module(x)
            if idx == 1:
                feature1 = x
        feature3 = x
        x = torch.cat((self.global_pooling(feature1), self.global_pooling(feature3)), 1)
        x = self.fc(x)
        return x


class MultiScale(nn.Module):
    def __init__(self, depth, downsampling):
        super(MultiScale, self).__init__()
        self.start_filts = 64
        self.input_dim = 1
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling

        self.final_outs = self.start_filts
        down_conv = []
        down_conv.append(BatchNorm(self.input_dim, self.start_filts))

        ins = 64
        for i in range(1, self.downsampling):
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
            ins = self.outs
        self.final_outs += self.outs
        self.down_convs = nn.Sequential(*down_conv)

        convs = []
        for _ in range(self.depth - self.downsampling):
            convs.append(BatchNorm(self.outs, self.outs))
        self.final_outs += self.outs
        self.convs = nn.Sequential(*convs)
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)

        self.fc = nn.Sequential(
            nn.Linear(self.final_outs, self.output_dim)
        )
        initialize_weights(self)
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))

    @staticmethod
    def avg_pooling(feature):
        return F.avg_pool2d(feature, kernel_size=feature.size()[2:]).squeeze()

    def forward(self, x):
        for idx, module in enumerate(self.down_convs, 1):
            x = module(x)
            if idx == 1:
                feature1 = x
            if idx == self.downsampling:
                feature2 = x
        for module in self.convs:
            x = module(x)
        feature3 = x
        x = torch.cat((self.avg_pooling(feature1), self.avg_pooling(feature2), self.avg_pooling(feature3)), 1)
        x = self.fc(x)
        return x


class ConvBatchNormLeaky(nn.Module):
    """
    model architecture: downsampling*(custom_conv-bn-leaky_relu) + (depth-m)*(custom_conv-bn-leaky_relu)
    note in the former downsampling is performed after every custom_conv layer.And in the latter input size is invariable.
    Thus the parameter downsampling means the times of downsampling.
    """

    def __init__(self, depth, downsampling):
        """
        :param depth: downsampling 2^depth
        """
        super(ConvBatchNormLeaky, self).__init__()
        self.start_filts = 64
        self.input_dim = 3
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling

        assert self.depth >= self.downsampling
        down_conv = []
        for i in range(self.downsampling):
            ins = self.input_dim if i == 0 else self.outs
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
        conv = []

        for _ in range(self.depth - self.downsampling):
            conv.append(BatchNorm(self.outs, self.outs))

        self.down_convs = nn.ModuleList(down_conv)
        self.conv = nn.ModuleList(conv)
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)
        self.fc = nn.Sequential(nn.Linear(self.outs, self.output_dim))

        initialize_weights(self)
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))
        print('the downsmapling times is %d' % self.downsampling)

    def forward(self, x):
        for module in self.down_convs:
            x = module(x)
        for module in self.conv:
            x = module(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze()
        x = self.fc(x)
        return x


def get_discriminator(dis_type, depth, dowmsampling):
    if dis_type == 'conv_bn_leaky_relu':
        print('use conv_bn_leaky_relu as discriminator and downsampling will be achieved for %d times.' % dowmsampling)
        d = ConvBatchNormLeaky(depth, dowmsampling)
    elif dis_type == 'multi_scale':
        print('use MultiScale as discriminator')
        d = MultiScale(depth, dowmsampling)
    elif dis_type == 'top_multi_scale':
        print('use TopMultiScale and use top layer rather than middle layer')
        d = TopMultiScale(depth, dowmsampling)
    elif dis_type == 'middle_multi_scale':
        print('use MiddleMultiScale and use middle layer rather than top layer')
        d = MiddleMultiScale(depth, dowmsampling)
    else:
        raise ValueError("parameter discriminator type must be in ['conv_bn_leaky_relu', 'multi_scale']")
    print(d)
    print('use discriminator with depth of %d and last custom_conv feature size is (%d,%d)' % (
        d.depth, d.last_size, d.last_size))
    return d


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # d = ConvBatchNormLeaky(7, 4)
    d = get_discriminator('multi_scale', 7, 4)
    # d = MultiScale(7, 4)
    import torchvision.models as models
    # d = models.resnet18(pretrained=False)
    print(d)
    tensor = torch.rand((2, 1, 128, 128))
    print(d(tensor))
    # print(d)
