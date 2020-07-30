#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/11 15:19
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn

# 本代码每次只迭代一个obj，每个obj单独进行反向传播，所以num_object一直为1
# 默认所有obj都在第一帧出现，后面才出现的obj不使用。

import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F
# from config import msra10k_path
from datasets_loader import ImageFolder
from utils.misc import AvgMeter, check_mkdir
# from model_efnet_v4 import SiamET
from models.model_EMN_SelectMem import EMN
from models.model_EMN import EMN as EMN_NoSelect
from models.model_STM import STM
from datasets_loader import DAVIS_MO_Test
from torch.backends import cudnn
from utils import joint_transforms
from utils.helpers import pad_divide_by
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
from train_utils import iou
from train_utils import toTensor
from train_utils import toCudaVariable
from train_utils import prepro_img
from train_utils import prepro_mask
from train_utils import train_3f1o
from train_utils import train_3f1o_NoSelect
from train_utils import palette

try:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
except:
    print('this is cy620')

# Initialize Horovod
hvd.init()
# Horovod Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

# args = {
#     'iter_num': 10,
#     'train_batch_size': 2,
#     'last_iter': 0,
#     'lr': 1e-6,
#     'lr_decay': 0.9,
#
#
#     'weight_decay': 5e-6,
#     'momentum': 0.9,
#     'snapshot': '120',
#     'loadepoch': '-1'
# }

# def update_args(args_load,args):
#     pass

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='EMN')
    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                        help='epoch to load model',
                        default='-1', type=str)
    parser.add_argument('--batchsize',dest='batchsize',
                        help='How large for batch size.',
                        default=2,type=int)
    parser.add_argument('--lr',dest='lr',
                        help='learning rate',
                        default=1e-5,type=float)
    parser.add_argument('--lr_decay', dest='lr_decay',
                        help='How much lr decay every iteration.',
                        default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight decay for optimizer.',
                        default=5e-6, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum for optimizer.',
                        default=0.9, type=float)
    parser.add_argument('--iter_num', dest='iter_num',
                        help='iter_num.',
                        default=12, type=int)
    parser.add_argument('--last_iter', dest='last_iter',
                        help='last_iter.',
                        default=0, type=int)
    args = parser.parse_args()
    return args

arg_load = parse_args()

if hvd.rank() == 0:
    writer = SummaryWriter('runs/main_EMN{}'.format(time.strftime("%m%d%H%M%S", time.localtime())))

cudnn.benchmark = True

torch.manual_seed(2018)
# torch.cuda.set_device(3)
root = './data/mask/DUTS-TR'

ckpt_path = './ckpt'
exp_name = 'mask_9'
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
msra10k_path = root

target_transform = transforms.ToTensor()
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomRotate(0)
])

# duts_set = ImageFolder('./data/data_duts.json', joint_transform=joint_transform, target_transform=target_transform)
# msra_set = ImageFolder('./data/data_msra.json', joint_transform=joint_transform, target_transform=target_transform)
# train_set = ConcatDataset([duts_set]+[msra_set])
davis2017_set = ImageFolder('./data/data_davis2017.json', joint_transform=joint_transform,
                            target_transform=target_transform)
tianchiyusai_set = ImageFolder('./data/data_tianchiyusai.json', joint_transform=joint_transform,
                            target_transform=target_transform)
yvos_set = ImageFolder('./data/data_yvos.json', joint_transform=joint_transform, target_transform=target_transform)
train_set = ConcatDataset(5 * [davis2017_set] + [tianchiyusai_set]+ [yvos_set])
# train_set = ConcatDataset([tianchiyusai_set])

# Horovod Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set, num_replicas=hvd.size(), rank=hvd.rank())

# train_loader = DataLoader(train_set, batch_size=arg_load.batchsize, num_workers=0, shuffle=True)
train_loader = DataLoader(train_set, batch_size=arg_load.batchsize, sampler=train_sampler)

DATA_ROOT = '/home/ldz/文档/DAVIS'
Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(16, 'val'), single_object=True)
Testloader = DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

##todo以下修正版
# criterion1 = nn.BCELoss().cuda()
##todo:以下原版
criterion1 = nn.CrossEntropyLoss().cuda()  # nn.BCEWithLogitsLoss().cuda()




def main():
    # net = nn.DataParallel(EMN())
    # net = STM()
    net = EMN()
    # net = EMN_NoSelect()
    net.cuda()

    # optimizer = optim.SGD(net.parameters(), lr=arg_load.lr,momentum=arg_load.momentum,weight_decay=arg_load.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=arg_load.lr, weight_decay=arg_load.weight_decay)

    if arg_load.loadepoch != '-1':
        print('Loading checkpoint @Epoch {}...'.format(arg_load.loadepoch))
        load_name = os.path.join(ckpt_path, exp_name, '{}.pth'.format(arg_load.loadepoch))
        checkpoint = torch.load(load_name)

        # # 相当于用''代替'module.'。
        # # 直接使得需要的键名等于期望的键名。
        # net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # del checkpoint

        # state = net.state_dict()
        # state.update({k.replace('module.', ''): v for k, v in checkpoint.items()})
        # net.load_state_dict(state)
        # del checkpoint,state

        state = net.state_dict()
        state.update({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        net.load_state_dict(state)
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint,state

        torch.cuda.empty_cache()  # 不清理有可能报error:137
    torch.cuda.empty_cache()
    print('  - complete!')


    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
    # Horovod Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)

    train(net, optimizer)


def train(net, optimizer):
    curr_iter = arg_load.last_iter

    for idx in range(0, arg_load.iter_num):
        total_loss_record = AvgMeter()
        loss0_record = AvgMeter()
        loss1_record = AvgMeter()
        loss2_record = AvgMeter()
        net.eval()
        if idx % 3 == 0 and idx != 0 and hvd.rank() ==0:
        # if 0 == 0 and hvd.rank() ==0:
            evalution(net, Testloader, writer, idx)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = arg_load.lr * (1 - float(curr_iter) / arg_load.iter_num) ** arg_load.lr_decay
            # optimizer.param_groups[0]['lr'] = arg_load.lr * (arg_load.lr_decay**(curr_iter) )
            # todo：以下为修改版
            color, imgs_path_temp, masks_path_temp, origin_shape_temp, img_len = data
            batch_size = imgs_path_temp[0].__len__()
            imgs_path = []
            masks_path = []
            origin_shape = []

            try:
                for b in range(batch_size):
                    idx_list = range(3) #range(img_len[b])
                    imgs_path.append([imgs_path_temp[l][b] for l in idx_list])
                    masks_path.append([masks_path_temp[l][b] for l in idx_list])
                    origin_shape.append((origin_shape_temp[0][b],origin_shape_temp[1][b]))
            except:
                print("shit! and continue!")
                continue
            img_ref = []
            mask_ref = []
            img_mid = []
            mask_mid = []
            img_target = []
            mask_target = []
            for b in range(batch_size):
                img_ref.append(prepro_img(imgs_path[b][0]))
                mask_ref.append(prepro_mask(masks_path[b][0],color[b]))

                img_mid.append(prepro_img(imgs_path[b][1]))
                mask_mid.append(prepro_mask(masks_path[b][1],color[b]))

                img_target.append(prepro_img(imgs_path[b][2]))
                mask_target.append(prepro_mask(masks_path[b][2],color[b]))

            img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target = toTensor(
                [img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target])
            img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target = toCudaVariable(
                [img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target])

            Fs = torch.cat([img_ref, img_mid, img_target], dim=1).permute(0, 2, 1, 3, 4)
            Ms_obj = torch.cat([mask_ref, mask_mid, mask_target], dim=2)
            Ms_None = torch.zeros_like(Ms_obj).expand(-1 ,9, -1, -1, -1)
            Ms = torch.cat([1 - Ms_obj, Ms_obj, Ms_None], dim=1)

            loss0, Es, pred = train_3f1o(model=net,
                                   optimizer=optimizer,
                                   loss_func= criterion1,
                                   Fs=Fs,
                                   Ms=Ms,
                                   num_frames=3,
                                   num_objects=1,
                                   Mem_every=1,
                                   Mem_number=None)
            total_loss = loss0

            # if iou(Es[:,1,1:],Ms[:,1,1:])  ==0 :
            #     print("hi")
            #     iou(Es[:, 1, 2], Ms[:, 1, 2])
            # ...log the running loss
            global_step = idx * len(train_loader) + i

            total_loss_record.update(total_loss.item(), batch_size)
            loss0_record.update(loss0.item(), batch_size)

            loss1_record.update(loss0.item(), batch_size)
            loss2_record.update(loss0.item(), batch_size)


            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [lossr %.5f], [lr %.13f][iou %.5f]' % \
                  (idx, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   optimizer.param_groups[0]['lr'], iou(Es[:, 1, 1:], Ms[:, 1, 1:]))
            print(log)
            if i % 50 == 0 and hvd.rank() == 0:
                # if hvd.rank() == 0:
                print('add point to Tensorboard')
                writer.add_scalar('training loss', float(total_loss), global_step)
                maskedimg = transforms.ToPILImage()(Fs[0, :, -1].detach().cpu()).convert('RGB')
                maskedimg.save('main_img_target.jpg')
                eyeout = transforms.ToPILImage()(Ms[0, 1, -1].detach().cpu())
                eyeout.putpalette(palette)
                eyeout.save(os.path.join('main_mask_target_gt.png'))
                output = Image.fromarray(pred[-1])
                output.putpalette(palette)
                output.save(os.path.join('main_output1.png'))

                writer.add_image('maskedimg', transforms.ToTensor()(maskedimg))
                writer.add_image('main_mask_target_gt', transforms.ToTensor()(eyeout))
                writer.add_image('main_output1', transforms.ToTensor()(output))
                writer.close()

                # Then we save models by steps [only 'rank==0' can run]
            if (global_step + 1) % 200 == 0 and hvd.rank() == 0:  # 避免测试时覆盖权重的问题
                # if  hvd.rank() == 0 :  # 避免测试时覆盖权重的问题
                save_state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'idx': idx}
                torch.save(
                    obj=save_state,
                    f=os.path.join(ckpt_path, exp_name, 'main_EMN_step_latest.pth')
                )
                print('After {} steps, we saved a step model.'.format(global_step))
                del save_state
                torch.cuda.empty_cache()  # 137

                # Then save two models [only 'rank==0' can run]
        if idx % 1 == 0 and hvd.rank() == 0:  # 避免测试时的覆盖权重问题
            save_state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'idx': idx}
            torch.save(
                obj=save_state,
                f=os.path.join(ckpt_path, exp_name, 'main_EMN_{}.pth'.format(idx))
            )
            torch.save(
                obj=save_state,
                f=os.path.join(ckpt_path, exp_name, 'main_EMN_latest.pth')
            )
            print('{}th idx ended, we saved 2 models.'.format(idx))
            del save_state
            torch.cuda.empty_cache()  # 137

        curr_iter += 1



############################  Only for Evalution  ############################

def evalution(net, Testloader, writer, global_step): #只能在hvd.rank()==0的情况下使用
    with torch.no_grad():
        loss_eval = 0
        iou_eval = 0
        print("we are now in eval:")
        for seq, V in enumerate(Testloader):
            Fs, Ms, num_objects, info = V
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0].item()
            print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
            [Fs,Ms] = toCudaVariable([Fs, Ms])
            pred, Es = Run_video(net, Fs, Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
            loss_eval += criterion1(Es, Ms).detach().cpu()
            iou_eval += iou(Ms, Es).detach().cpu()
        loss_eval /= (seq + 1)
        iou_eval /= (seq + 1)
        print('loss_eval:{}'.format(loss_eval))
        # if hvd.rank() == 0 :
        writer.add_scalar('evalution loss', float(loss_eval), global_step)
        writer.close()
        print('iou_eval:{}'.format(iou_eval))
        print("eval is finished")

def Run_video(model, Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    msk_shape = (Ms[0, 1, 0].size()[0], Ms[0, 1, 0].size()[1])
    [img_first, mask_first], pad = pad_divide_by([Fs[:, :, 0], Ms[0, 1, 0]], 16, msk_shape)
    imsk_1st_qtr = transforms.ToTensor()(transforms.ToPILImage()((mask_first * img_first)[0].detach().cpu()).resize(
        (int(msk_shape[0] / 4), int(msk_shape[1] / 4)))).unsqueeze(0).cuda()
    # for t in tqdm.tqdm(range(1, num_frames)):
    for t in range(1, num_frames):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

        if t - 1 == 0:  #
            this_keys, this_values = prev_key, prev_value  # only prev memory
        else:
            # this_keys = torch.cat([keys, prev_key], dim=3)
            # this_values = torch.cat([values, prev_value], dim=3)

            this_keys = (t - 1) / t * keys + 1 / t * prev_key
            this_values = (t - 1) / t * values + 1 / t * prev_value

        # segment
        with torch.no_grad():
            logit = model(Fs[:, :, t], imsk_1st_qtr, this_keys, this_values, torch.tensor([num_objects]))
        Es[:, :, t] = F.softmax(logit, dim=1)

        # update
        if t - 1 in to_memorize:
            keys, values = this_keys, this_values

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es
############################   Evalution_code ends  ############################


if __name__ == '__main__':
    main()
