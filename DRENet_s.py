import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from SSA import shunted_t


stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512




class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AMFW(nn.Module):
    def __init__(self, channel, M=2, k_size=3):
        super().__init__()
        self.M = M
        self.channel = channel
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size, channel, _, _ = x1.shape
        feats = torch.cat([x1, x2], dim=1)
        feats = feats.view(batch_size, self.M, self.channel, feats.shape[2], feats.shape[3])

        feats_S = torch.sum(feats, dim=1)
        feats_G = self.gap(feats_S)

        feats_G = feats_G.squeeze(-1).transpose(-1, -2)
        attention_vectors = [conv(feats_G).transpose(-1, -2).unsqueeze(-1) for conv in self.convs]

        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_o = torch.sum(feats * attention_vectors.expand_as(feats), dim=1)

        return feats_o

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit

    Parameter:
        'normalize' is not necessary if the input size is fixed
    """

    def __init__(self, num_in, num_mid,
                 normalize=False):
        super(GloRe_Unit, self).__init__()

        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in, eps=1e-04)  # should be zero initialized

    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out
class AFM(nn.Module):
    def __init__(self, in_channel):
        super(AFM, self).__init__()
        self.conv = DSConv3x3(in_channel, in_channel)
        self.gcn = GloRe_Unit(in_channel, in_channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x_conv = self.conv(x)
        x_gcn = self.gcn(x)
        x_out = self.sigmoid(x_gcn) * x_conv + x_conv
        return x_out


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                # nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)



class model_student(nn.Module):
    def __init__(self,):
        super(model_student, self).__init__()
        # Backbone model
        self.rgb = shunted_t(pretrained=True)
        self.depth = shunted_t(pretrained=True)

        self.amfw1 = AMFW(stage1_channel)
        self.amfw2 = AMFW(stage2_channel)
        self.amfw3 = AMFW(stage3_channel)
        self.amfw4 = AMFW(stage4_channel)


        self.afm1 = AFM(stage1_channel)
        self.afm2 = AFM(stage2_channel)
        self.afm3 = AFM(stage3_channel)
        self.afm4 = AFM(stage4_channel)


        self.conv_stage1to2 = convbnrelu(stage1_channel, stage2_channel)
        self.conv_stage2to3 = convbnrelu(stage2_channel, stage3_channel)
        self.conv_stage3to4 = convbnrelu(stage3_channel, stage4_channel)

        self.conv_stage2to1 = convbnrelu(stage2_channel, stage1_channel)
        self.conv_stage3to2 = convbnrelu(stage3_channel, stage2_channel)
        self.conv_stage4to3 = convbnrelu(stage4_channel, stage3_channel)
        self.conv_stage4to2 = convbnrelu(stage4_channel, stage2_channel)
        self.conv_stage4to1 = convbnrelu(stage4_channel, stage1_channel)
        self.conv_stage3to1 = convbnrelu(stage3_channel, stage1_channel)

        #调整HW，如果需要加倍，用上采样H->2H W->2W
        self.upsample64 = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 调整HW，如果需要减半，用池化2H->H 2W->W
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(4, stride=4, ceil_mode=True)
        self.pool8 = nn.MaxPool2d(8, stride=8, ceil_mode=True)




        self.Head1 = SalHead(stage1_channel)
        self.Head2 = SalHead(stage2_channel)
        self.Head3 = SalHead(stage3_channel)
        self.Head4 = SalHead(stage4_channel)



    def forward(self, x_rgb, x_depth):
        rgb_list = self.rgb(x_rgb)
        rgb_1 = rgb_list[0]
        rgb_2 = rgb_list[1]
        rgb_3 = rgb_list[2]
        rgb_4 = rgb_list[3]

        x_depth = torch.cat([x_depth, x_depth, x_depth], dim=1)
        depth_list = self.depth(x_depth)
        depth_1 = depth_list[0]
        depth_2 = depth_list[1]
        depth_3 = depth_list[2]
        depth_4 = depth_list[3]

        amfw_out1 = self.amfw1(rgb_1, depth_1)
        amfw_out2 = self.amfw2(rgb_2, depth_2)
        amfw_out3 = self.amfw3(rgb_3, depth_3)
        amfw_out4 = self.amfw4(rgb_4, depth_4)
        # amfw_out1 = rgb_1+depth_1
        # amfw_out2 = rgb_2+depth_2
        # amfw_out3 = rgb_3+depth_3
        # amfw_out4 = rgb_4+depth_4


        afm_out4 = self.afm4(amfw_out4)
        afm_out3 = self.afm3(self.upsample2(self.conv_stage4to3(afm_out4)) + amfw_out3)
        afm_out2 = self.afm2(self.upsample2(self.conv_stage3to2(afm_out3)) + amfw_out2)
        afm_out1 = self.afm1(self.upsample2(self.conv_stage2to1(afm_out2)) + amfw_out1)
        # afm_out4 = amfw_out4
        # afm_out3 = self.upsample2(self.conv_stage4to3(afm_out4)) + amfw_out3
        # afm_out2 = self.upsample2(self.conv_stage3to2(afm_out3)) + amfw_out2
        # afm_out1 = self.upsample2(self.conv_stage2to1(afm_out2)) + amfw_out1


        out4 = self.Head4(afm_out4)
        out3 = self.Head3(afm_out3)
        out2 = self.Head2(afm_out2)
        out1 = self.Head1(afm_out1)

        # print("out1", out1.shape)
        # print("out2", out2.shape)
        # print("out3", out3.shape)
        # print("out4", out4.shape)
        out1 = self.upsample4(out1)
        out2 = self.upsample8(out2)
        out3 = self.upsample16(out3)
        out4 = self.upsample32(out4)
        # return out1, out2, out3, out4
        return out1, out2, out3, out4,\
                amfw_out1, amfw_out2, amfw_out3, amfw_out4,\
                afm_out1, afm_out2, afm_out3, afm_out4



if __name__ == '__main__':
    input_rgb = torch.randn(2, 3, 256, 256)
    input_depth = torch.randn(2, 1, 256, 256)
    net = model_student()
    out = net(input_rgb, input_depth)
    print("out1", out[0].shape)
    print("out2", out[1].shape)
    print("out3", out[2].shape)
    # print("out4", out[3].shape)
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 1, 256, 256)
    model = model_student()
    from FLOP import CalParams

    CalParams(model, a, b)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # input_rgb = torch.randn(2, 128, 32, 32)
    # input_depth = torch.randn(2, 128, 32, 32)
    # input_depth = torch.randn(2, 128, 32, 32)
    #
    # net = DSA(128)
    # out = net(input_rgb, input_depth, input_depth)
    # print("out", out.shape)