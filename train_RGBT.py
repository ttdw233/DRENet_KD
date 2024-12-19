import torch
from torch import nn
# from RGBT_dataprocessing_CNet import trainData, valData
from RGBT_dataprocessing_CNet import trainData, valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import Loss.lovasz_losses as lovasz
import pytorch_iou
from DRENet_t import model_teacher
from DRENet_s import model_student
import torchvision
import time
import os
import shutil
from log import get_logger
import yaml
# from  Loss.Binary_Dice_loss import BinaryDiceLoss
# from Loss.Focal_loss import sigmoid_focal_loss
import matplotlib.pyplot as plt
import pytorch_fm

# torch.autograd.set_detect_anomaly(True)
#输出模型的参数量
def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params +=p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))

#损失函数
IOU = pytorch_iou.IOU(size_average = True).cuda()

#交叉熵损失函数
class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                lossall = self.nll_lose(inputs, targets)
        losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 4
HW =256
#修改图片尺寸，1、dataprocessing_cnet 2、这里 3、下面的除数
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)

test_dataloader = DataLoader(valData,batch_size=batchsize,shuffle=True,num_workers=4)

# path = "/home/yuride/Documents/rail_frame/HRTransNet/config/hrt_base.yaml"
# config = yaml.load(open(path, 'r'), yaml.SafeLoader)['MODEL']['HRT']
net =model_1()
# net =model_2()
# net.load_pre('/home/yuride/Documents/document/model/AMENet/uniformer_small_tl_384 (1).pth')
# net.load_state_dict(torch.load('/home/hjk/文档/third_model_GCN/Pth5/Fourth_new_wanzhen_SOD_2023_03_29_21_46_best.pth'))
net = net.cuda()

################################################################################################################
model = 'net1' + time.strftime("_%Y_%m_%d_%H_%M")
print_network(net, model)
################################################################################################################
bestpath = '/media/pc12/data/hyt/rail_frame/train_test1/Pth_up/' + model + '_best.pth'
lastpath = '/media/pc12/data/hyt/rail_frame/train_test1/Pth_up/' + model + '_last.pth'
################################################################################################################
criterion1 = BCELOSS().cuda()
criterion2 = BCELOSS().cuda()
criterion3 = BCELOSS().cuda()
criterion4 = BCELOSS().cuda()
criterion5 = BCELOSS().cuda()
criterion6 = BCELOSS().cuda()
criterion7 = BCELOSS().cuda()
criterion8 = BCELOSS().cuda()

# focaloss = sigmoid_focal_loss().cuda()
# diceloss = BinaryDiceLoss().cuda()

criterion_val = BCELOSS().cuda()
#dice损失
def dice_loss(pred, mask):
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

#蒸馏损失
def kd_loss(pred, mask):
    b1, c1, h1 ,w1 = pred.shape
    b2, c2, h2, w2 = mask.shape
    pred_reshape = pred.reshape(b1, c1, -1)
    pred_tr = pred_reshape.permute(0, 2, 1)
    mask_reshape = mask.reshape(b2, c2, -1)
    mask_tr = mask_reshape.permute(0, 2, 1)
    mul_pred = torch.bmm(pred_tr, pred_reshape)
    mul_mask = torch.bmm(mask_tr, mask_reshape)
    softmax_pred = F.softmax(mul_pred/np.sqrt(c1), dim=0)
    logsoftmax = nn.LogSoftmax(dim=0)
    softmax_mask = logsoftmax(mul_mask/np.sqrt(c2))
    loss = (torch.sum(- softmax_pred * softmax_mask))/w1/h1
    return loss

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
floss = pytorch_fm.FLoss()

################################################################################################################
lr_rate = 1e-4#学习率
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)#优化方式
################################################################################################################

best = [10]#在10个里面选最好的
step = 0
mae_sum = 0
best_mae = 1
best_epoch = 0

#存训练日志
logdir = f'Run_up/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 150
################################################################################################################

#训练日志
logger.info(f'Epochs:{epochs}  Batchsize:{batchsize} HW:{HW}')
#迭代训练
for epoch in range(epochs):
    mae_sum = 0
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']


    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        # image = Variable(sample['RGB'].float().cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())
        # background1 = 1 - label

        #模型的参数更新优化
        optimizer.zero_grad()

        out1, out2, out3, out4 = net(image, depth)#左边是模型的输出，右边是模型的输入
        # s12, s22, s32, s42, r1_new, d1_new, r4_new, d4_new, s1_new = net(depth, depth)
        # out1, out2, out3, out4, r1, d1, r4, d4, s1_new, Z4, S4, rd4 = net(image, depth)

        #上面有几个输出，就做几个sigmoid
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        out4 = torch.sigmoid(out4)
        # out5 = torch.sigmoid(out5)

        # fore_loss1 = criterion1(s3_out, label) + IOU(s3_out, label)
        # fore_loss2 = criterion1(s2_out, label) + IOU(s2_out, label)
        # fore_loss3 = criterion1(s1_out, label) + IOU(s1_out, label)
        # fore_loss4 = criterion1(s4_out, label) + IOU(s4_out, label)

        #几个输出就写几个loss
        loss1 = criterion1(out1, label) + IOU(out1, label) #+ criterion1(torch.sigmoid(edg1), bound)
        loss2 = criterion1(out2, label) + IOU(out2, label) #+ criterion1(torch.sigmoid(edg2), bound)
        loss3 = criterion1(out3, label) + IOU(out3, label) #+ criterion1(torch.sigmoid(edg3), bound)
        loss4 = criterion1(out4, label) + IOU(out4, label) #+ criterion1(torch.sigmoid(edg4), bound)

        #
        # # kd_loss1 = dice_loss(r1_new, s1_new)
        # # kd_loss2 = dice_loss(d1_new, s1_new)
        # # kd_loss3 = dice_loss(r1_new, r4_new)
        # # kd_loss4 = dice_loss(d1_new, d4_new)
        #
        #

        loss_total = loss1 + loss2 + loss3 + loss4

        # loss_total = fore_loss1 + fore_loss2 + fore_loss3 + fore_loss4 \
        #              + kd_loss1 + kd_loss2 + kd_loss3 + kd_loss4
        # loss_total = loss + iou_loss


        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), loss_total))
        loss_total.backward()#反向传播
        optimizer.step()
        train_loss = loss_total.item() + train_loss


    net = net.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            depthVal = Variable(sampleTest['depth'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())
            boundVal = Variable(sampleTest['bound'].float().cuda())

            out1 = net(imageVal, depthVal)
            #out1 = net(imageVal)

            #out如果不止一个就用列表接收，最终输出就用out1[0]表示，如果输出只有一个，那么out1就代表最终输出
            out = F.sigmoid(out1[0])
            # out = F.sigmoid(out1)

            loss = criterion_val(out, labelVal)
            maeval = torch.sum(torch.abs(labelVal - out)) / (256.0*256.0)

            print('===============', j, '===============', loss.item())
    #
    #         # if j==34:
    #         #     out=out[4].cpu().numpy()
    #         #     edge = edge[4].cpu().numpy()
    #         #     out = out.squeeze()567
    #         #     edge = edge.squeeze()
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/out.png', out,cmap='gray')
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/edge1.png', edge,cmap='gray')
    #
            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 1500:.8f} valloss:{eval_loss / 362:.8f} || '
        f'valmae:{mae / 362:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 362) <= min(best):
        best.append(mae / 362)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae, min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')














