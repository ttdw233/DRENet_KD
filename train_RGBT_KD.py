import torch
from torch import nn
import copy
# from RGBT_dataprocessing_CNet import trainData, valData
from train_test1.RGBT_dataprocessing_CNet import trainData, valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import numpy as np
# import Loss.lovasz_losses as lovasz
import pytorch_iou
import pytorch_fm
# from  Self_KD.Ablation_self_kd_model_seven import test_model
from DRENet_t import model_teacher
from DRENet_s import model_student
from loss import KLDLoss, hcl, dice_loss
from kdloss.contraloss import ContrastLoss
from kdloss.kdloss.sobel import FilterLoss


import torchvision
import torch.nn.functional as F
import time
import os
import shutil
from log import get_logger



def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))


IOU = pytorch_iou.IOU(size_average=True).cuda()
floss = pytorch_fm.FLoss()

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
batchsize = 6
HW = 256
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)

test_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=True, num_workers=4)

teacher_net = model_teacher()
teacher_net.load_state_dict(torch.load('/media/pc12/data/hyt/rail_frame/train_test1/Pth/GOODNET_t_2024_06_01_10_59_best.pth'))
net = model_student()

net = net.cuda()
teacher_net = teacher_net.cuda()
################################################################################################################
model = 'KD_model_T_S_kd_wo_con_2' + time.strftime("_%Y_%m_%d_%H_%M")

print_network(net, model)
################################################################################################################
bestpath = './KD_Pth/' + model + '_best.pth'
lastpath = './KD_Pth/' + model + '_last.pth'
################################################################################################################

stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512

stage1_HW = 64
stage2_HW = 32
stage3_HW = 16
stage4_HW = 8

criterion1 = BCELOSS().cuda()

fliter_loss1 = FilterLoss(stage1_channel).cuda()
fliter_loss2 = FilterLoss(stage2_channel).cuda()
fliter_loss3 = FilterLoss(stage3_channel).cuda()
fliter_loss4 = FilterLoss(stage4_channel).cuda()
con_loss1 = ContrastLoss(4, stage1_channel).cuda()
con_loss2 = ContrastLoss(4, stage2_channel).cuda()
con_loss3 = ContrastLoss(4, stage3_channel).cuda()
con_loss4 = ContrastLoss(4, stage4_channel).cuda()
# graph_loss1 = GraphContrastDistill(stage1_channel).cuda()
# graph_loss2 = GraphContrastDistill(stage2_channel).cuda()
# graph_loss3 = GraphContrastDistill(stage3_channel).cuda()
# graph_loss4 = GraphContrastDistill(stage4_channel).cuda()



# criterion2 = ATLoss().cuda()
# sp_loss = Similarity().cuda()
# skd_loss = self_kd_loss().cuda()
kld_loss = KLDLoss().cuda()


criterion_val = BCELOSS().cuda()
################################################################################################################
lr_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)
################################################################################################################

best = [10]                    
step = 0                       
mae_sum = 0                    
best_mae = 1                   
best_epoch = 0
running_loss_pre = 0.0

logdir = f'KD_run/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 200
################################################################################################################

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize} HW:{HW}')
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
    teacher_net = teacher_net.eval()
    prec_time = datetime.now()

    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())
        # print('image', image.shape)
        # print('depth', depth.shape)
        # image = image.unsqueeze(2)
        # depth = depth.unsqueeze(2)
        optimizer.zero_grad()

        with torch.no_grad():
            T_out1, T_out2, T_out3, T_out4, T_dgf1, T_dgf2, T_dgf3, T_dgf4, T_mcam1, T_mcam2, T_mcam3, T_mcam4, \
                = teacher_net(image, depth)

        S_out1, S_out2, S_out3, S_out4, S_dgf1, S_dgf2, S_dgf3, S_dgf4, S_mcam1, S_mcam2, S_mcam3, S_mcam4, \
                = net(image, depth)

        S_out1 = torch.sigmoid(S_out1)
        S_out2 = torch.sigmoid(S_out2)
        S_out3 = torch.sigmoid(S_out3)
        S_out4 = torch.sigmoid(S_out4)


        loss1 = criterion1(S_out1, label) + IOU(S_out1, label)
        loss2 = criterion1(S_out2, label) + IOU(S_out2, label)
        loss3 = criterion1(S_out3, label) + IOU(S_out3, label)
        loss4 = criterion1(S_out4, label) + IOU(S_out4, label)

        loss_label = loss1 + loss2 + loss3 + loss4


        con_loss_encoder1 = con_loss1(S_dgf1, T_dgf1)
        con_loss_encoder2 = con_loss2(S_dgf2, T_dgf2)
        con_loss_encoder3 = con_loss3(S_dgf3, T_dgf3)
        con_loss_encoder4 = con_loss4(S_dgf4, T_dgf4)

        loss_con = con_loss_encoder1 + con_loss_encoder2 + con_loss_encoder3 + con_loss_encoder4


        loss_filter1 = fliter_loss1(S_mcam1, T_mcam1)
        loss_filter2 = fliter_loss2(S_mcam2, T_mcam2)
        loss_filter3 = fliter_loss3(S_mcam3, T_mcam3)
        loss_filter4 = fliter_loss4(S_mcam4, T_mcam4)

        loss_filter = loss_filter1 + loss_filter2 + loss_filter3 + loss_filter4


        loss_predict1 = dice_loss(S_out1, T_out1).cuda()
        # loss_predict2 = dice_loss(S_out2, T_out2).cuda()
        # loss_predict3 = dice_loss(S_out3, T_out3).cuda()
        # loss_predict4 = dice_loss(S_out4, T_out4).cuda()
        loss_predict = loss_predict1

        loss_total = loss_label + loss_predict + loss_filter + loss_con
        # loss_total = loss1 + loss2 + loss3 + loss4 + loss5
        # loss_total = loss + iou_loss

        time = datetime.now()

        if i % 10 == 0:
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), loss_label))
        loss_total.backward()
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
            # bound = Variable(sampleTest['bound'].float().cuda())

            # out1 = net(imageVal)
            out1 = net(imageVal, depthVal)

            out1 = torch.sigmoid(out1[0])
            out = out1
            loss = criterion_val(out, labelVal)

            maeval = torch.sum(torch.abs(labelVal - out)) / (256.0*256.0)

            print('===============', j, '===============', loss.item())
    #
    #         # if j==34:
    #         #     out=out[4].cpu().numpy()
    #         #     edge = edge[4].cpu().numpy()
    #         #     out = out.squeeze()
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














