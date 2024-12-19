import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from GOODNET.ifvd import CriterionIFV

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

class Sobel_filter(nn.Module):
    def __init__(self, in_channel):
        super(Sobel_filter, self).__init__()
        soble_kernel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                       [-2.0, 0.0, 2.0],
                                       [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        soble_kernel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                       [0.0, 0.0, 0.0],
                                       [1.0, 2.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        soble_kernel_x = soble_kernel_x.repeat(in_channel, 1, 1, 1)
        soble_kernel_y = soble_kernel_y.repeat(in_channel, 1, 1, 1)

        self.conv_x = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel,
                                bias=False)
        self.conv_y = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel,
                                bias=False)

        self.conv_x.weight.data = soble_kernel_x
        self.conv_y.weight.data = soble_kernel_y

        for param in self.conv_x.parameters():
            param.requires_grad = False
        for param in self.conv_y.parameters():
            param.requires_grad = False

    def forward(self, img):
        soble_x = self.conv_x(img)
        soble_y = self.conv_y(img)

        sobel_combined = torch.sqrt(soble_x ** 2 + soble_y ** 2 + 1e-6)

        return sobel_combined

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
class FilterLoss(nn.Module):
    def __init__(self, out_channel):
        super(FilterLoss, self).__init__()

        self.att_s_s = nn.Sequential(
            convbnrelu(out_channel, out_channel),
            nn.Sigmoid()
        )
        self.att_t_s = nn.Sequential(
            convbnrelu(out_channel, out_channel),
            nn.Sigmoid()
        )
        self.att_t_g = nn.Sequential(
            convbnrelu(out_channel, out_channel),
            nn.Sigmoid()
        )

        self.att_s_g = nn.Sequential(
            convbnrelu(out_channel, out_channel),
            nn.Sigmoid()
        )
        self.sobel = Sobel_filter(out_channel)
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=3)


    def forward(self, x_S, x_T):

        x_s_gaussian = self.gaussian(x_S)
        x_t_gaussian = self.gaussian(x_T)
        x_s_sobel = self.sobel(x_S)
        x_t_sobel = self.sobel(x_T)

        x_s_fuse = x_s_gaussian * self.att_s_s(x_s_gaussian) + x_s_sobel * self.att_s_g(x_s_sobel) + x_S
        x_t_fuse = x_t_gaussian * self.att_t_s(x_t_gaussian) + x_t_sobel * self.att_t_g(x_t_sobel) + x_T

        x_s_loss = hcl(x_s_sobel, x_t_sobel)
        x_g_loss = hcl(x_s_gaussian, x_t_gaussian)
        x_fuse_loss = hcl(x_s_fuse, x_t_fuse)

        loss = x_s_loss + x_g_loss + x_fuse_loss
        return loss

