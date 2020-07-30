#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 下午8:07
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : train_utils.py

import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
# from config import msra10k_path
# from model_efnet_v4 import SiamET
from utils.helpers import pad_divide_by
palette = Image.open('/home/ldz/文档/DAVIS/Annotations/480p/blackswan/00000.png').getpalette()
size_640_384 = (640, 384)  # Image是先宽后高
γ = 0.2
τ = 0

def crop_obj(mask_first, img_first, obj_msk_shape):
    h_min = []
    h_max = []
    w_min = []
    w_max = []
    batch_size = mask_first.shape[0]
    for i in range(batch_size):
        try:
            h_min.append(torch.min(torch.where(mask_first[i])[0]))
            h_max.append(torch.max(torch.where(mask_first[i])[0]))
            w_min.append(torch.min(torch.where(mask_first[i])[1]))
            w_max.append(torch.max(torch.where(mask_first[i])[1]))
        except:
            h_min.append(0)
            h_max.append(mask_first[i].shape[0])
            w_min.append(0)
            w_max.append(mask_first[i].shape[1])

    img_obj = [(mask_first * img_first.permute(1,0,2,3)).permute(1,0,2,3)[i][:, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1] for i in range(batch_size)]
    # img_obj = transforms.ToTensor()(transforms.ToPILImage()(img_obj.detach().cpu()).resize(obj_msk_shape)).unsqueeze(0).cuda()
    img_obj_temp = torch.zeros([batch_size,3,obj_msk_shape[1],obj_msk_shape[0]])# 先H后W
    for i in range(batch_size):
        img_obj_temp[i] = transforms.ToTensor()(transforms.ToPILImage()(img_obj[i].detach().cpu()).resize(obj_msk_shape))
    return img_obj_temp.cuda()

def iou(target, pred):
    target = torch.where(target > 0.5, torch.ones_like(target), torch.zeros_like(target)).to(torch.uint8)
    pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred)).to(torch.uint8)
    intersection = torch.sum((target + pred) == 2).to(torch.float)
    union = torch.sum((target + pred) > 0).to(torch.float) + 1e-11
    iou_ans = intersection / union
    return iou_ans

def toCudaVariable(xs, volatile=False):
    if torch.cuda.is_available():
        return [Variable(x.cuda(), volatile=volatile) for x in xs]
    else:
        return [Variable(x, volatile=volatile) for x in xs]


def toTensor(xs):
    return [torch.tensor(x) for x in xs]

def prepro_img(img_path, size=size_640_384):
    img_path= img_path.replace('Youtube_VOS','YouTube-VOS')
    img = np.array(Image.open(img_path).resize(size).convert("RGB")) / 255.0
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
    return img


def prepro_mask(mask_path, color, size=size_640_384):
    mask_path = mask_path.replace('Youtube_VOS', 'YouTube-VOS')
    mask = np.expand_dims(
        np.array(Image.open(mask_path).resize(size).convert('L'), dtype=np.uint8), axis=0)
    mask = np.expand_dims(chose_obj(mask, color), axis=0).astype(np.float32)
    return mask

def chose_obj(mask, color):
    return (mask == np.ones_like(mask) * int(color)).astype(np.uint8)


def train_3f1o(model, optimizer,loss_func, Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    # msk_shape = (Ms[0, 1, 0].size()[0], Ms[0, 1, 0].size()[1])
    # obj_msk_shape = (int(msk_shape[1] / 4), int(msk_shape[0] / 4))  # Image先宽后高
    # [img_first, mask_first], pad = pad_divide_by([Fs[:, :, 0], Ms[:, 1, 0]], 16, msk_shape)
    # imsk_1st_qtr = crop_obj(mask_first, img_first, obj_msk_shape)
    with torch.autograd.set_detect_anomaly(True):
        for t in range(1, num_frames):
            #keys,values 是过去的所有
            #prev_key是上一帧的信息
            #this_key是当前帧的信息

            # memorize
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

            if t - 1 == 0:  #
                this_keys, this_values = prev_key, prev_value  # only prev memory
            else:
            #     this_keys = torch.cat([keys, prev_key], dim=3)
            #     this_values = torch.cat([values, prev_value], dim=3)
            #     # 没有到to_memorize之前，keys一直是上个轮回的this_key,但prev在不断更新，所以this_key实际上是在一直变动的
            #
            #     # this_keys = (t - 1) / t * keys + 1/t * prev_key
            #     # this_values = (t - 1) / t * values + 1/t * prev_value
                this_keys   = (sim*keys.transpose(0,-1)   +(1-sim)*prev_key.transpose(0,-1)).transpose(0,-1)
                this_values = (sim*values.transpose(0,-1) +(1-sim)*prev_value.transpose(0,-1)).transpose(0,-1)

            # segment
            # logit = model(Fs[:, :, t], imsk_1st_qtr, this_keys, this_values, torch.tensor([num_objects]))
            logit,Q_propagate,sim = model(Fs[:, :, t], this_keys, this_values, torch.tensor([num_objects]))

            Uk_i = torch.where(Q_propagate < τ, torch.ones_like(Q_propagate), torch.zeros_like(Q_propagate))
            Np = (Q_propagate.shape[1] * Q_propagate.shape[2] * Q_propagate.shape[3])
            Qk_i = torch.sum(Uk_i, dim=(1, 2, 3)) / Np
            whether_key_frame = Qk_i > γ

            # # todo:使用上一帧作为参考begin
            # [img_last, mask_last], pad = pad_divide_by([Fs[:, :, t], logit[:, 1]], 16, msk_shape)
            # imsk_1st_qtr = crop_obj((mask_last>0.5).to(torch.float32), img_last, obj_msk_shape)
            # # todo:使用上一帧作为参考end

            # update
            Es[:, :, t] = F.softmax(logit, dim=1)

            loss_CE = loss_func(logit, torch.argmax(Ms[:, :, t], dim=1))
            optimizer.zero_grad()
            if t != num_frames - 1:
                loss_CE.backward(retain_graph=True)
            else:
                loss_CE.backward()
            optimizer.step()

            # if t - 1 in to_memorize:
            #     keys, values = this_keys, this_values
            if t - 1 == 0:
                keys, values = this_keys, this_values
            else:
                for b, whether in enumerate(whether_key_frame):
                    if whether == True:  # key frame
                        keys[b], values[b] = this_keys[b], this_values[b]
                        print(f'Whether{b} is True')
                    elif whether == False:  # key frame
                        pass

            del logit, prev_key, prev_value, this_keys, this_values
            torch.cuda.empty_cache()  # 不清理有可能报error:137

    pred = np.argmax(Es[0].detach().cpu().numpy(), axis=0).astype(np.uint8)
    return loss_CE.detach().cpu(), Es, pred

def train_3f1o_NoSelect(model, optimizer,loss_func, Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    # msk_shape = (Ms[0, 1, 0].size()[0], Ms[0, 1, 0].size()[1])
    # obj_msk_shape = (int(msk_shape[1] / 4), int(msk_shape[0] / 4))  # Image先宽后高
    # [img_first, mask_first], pad = pad_divide_by([Fs[:, :, 0], Ms[:, 1, 0]], 16, msk_shape)
    # imsk_1st_qtr = crop_obj(mask_first, img_first, obj_msk_shape)
    with torch.autograd.set_detect_anomaly(True):
        for t in range(1, num_frames):
            #keys,values 是过去的所有
            #prev_key是上一帧的信息
            #this_key是当前帧的信息

            # memorize
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

            if t - 1 == 0:  #
                this_keys, this_values = prev_key, prev_value  # only prev memory
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)
                # 没有到to_memorize之前，keys一直是上个轮回的this_key,但prev在不断更新，所以this_key实际上是在一直变动的
            #
            #     # this_keys = (t - 1) / t * keys + 1/t * prev_key
            #     # this_values = (t - 1) / t * values + 1/t * prev_value


            # segment
            # logit = model(Fs[:, :, t], imsk_1st_qtr, this_keys, this_values, torch.tensor([num_objects]))
            logit= model(Fs[:, :, t], this_keys, this_values, torch.tensor([num_objects]))


            # # todo:使用上一帧作为参考begin
            # [img_last, mask_last], pad = pad_divide_by([Fs[:, :, t], logit[:, 1]], 16, msk_shape)
            # imsk_1st_qtr = crop_obj((mask_last>0.5).to(torch.float32), img_last, obj_msk_shape)
            # # todo:使用上一帧作为参考end

            # update
            Es[:, :, t] = F.softmax(logit, dim=1)

            loss_CE = loss_func(logit, torch.argmax(Ms[:, :, t], dim=1))
            optimizer.zero_grad()
            if t != num_frames - 1:
                loss_CE.backward(retain_graph=True)
            else:
                loss_CE.backward()
            optimizer.step()

            if t - 1 in to_memorize:
                keys, values = this_keys, this_values


            del logit, prev_key, prev_value, this_keys, this_values
            torch.cuda.empty_cache()  # 不清理有可能报error:137
    pred = np.argmax(Es[0].detach().cpu().numpy(), axis=0).astype(np.uint8)

    return loss_CE.detach().cpu(), Es, pred