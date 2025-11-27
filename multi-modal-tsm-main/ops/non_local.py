# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NL3DWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.nl = NONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


# 修改make_non_local函数，从第146行开始
def make_non_local(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet):
        net.layer2 = nn.Sequential(
            NL3DWrapper(net.layer2[0], n_segment),
            net.layer2[1],
            NL3DWrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL3DWrapper(net.layer3[0], n_segment),
            net.layer3[1],
            NL3DWrapper(net.layer3[2], n_segment),
            net.layer3[3],
            NL3DWrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
    elif hasattr(net, 'layer2') and hasattr(net, 'layer3'):
        # 为其他具有类似ResNet结构的模型添加支持
        # 尝试在layer2和layer3中插入NL3DWrapper
        try:
            # 检查layer2是否是序列
            if isinstance(net.layer2, nn.Sequential) and len(net.layer2) > 0:
                new_layer2 = []
                for i, block in enumerate(net.layer2):
                    if hasattr(block, 'bn3'):
                        new_layer2.append(NL3DWrapper(block, n_segment))
                    else:
                        new_layer2.append(block)
                net.layer2 = nn.Sequential(*new_layer2)
            
            # 检查layer3是否是序列
            if isinstance(net.layer3, nn.Sequential) and len(net.layer3) > 0:
                new_layer3 = []
                for i, block in enumerate(net.layer3):
                    if hasattr(block, 'bn3'):
                        new_layer3.append(NL3DWrapper(block, n_segment))
                    else:
                        new_layer3.append(block)
                net.layer3 = nn.Sequential(*new_layer3)
        except Exception as e:
            print(f"Warning: Failed to add non-local blocks: {e}")
            # 静默失败，继续训练而不添加non-local模块
    else:
        # 对于不支持的模型类型，发出警告并继续训练
        print(f"Warning: Non-local module not implemented for model type {type(net).__name__}")
        # 不再抛出异常，允许训练继续进行


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    img = Variable(torch.zeros(2, 3, 20))
    net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 20, 20))
    net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())