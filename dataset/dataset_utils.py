#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 下午12:09
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : hi.py
import os
import os.path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def pad_size_to_32times(oh_masks,raw_frames):
    # padding size to be divide by 32
    nf, h, w, _ = oh_masks.shape
    new_h = h + 32 - h % 32
    new_w = w + 32 - w % 32
    # print(new_h, new_w)
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
    pad_masks = np.pad(oh_masks, ((0, 0), (lh, uh), (lw, uw), (0, 0)), mode='constant')
    pad_frames = np.pad(raw_frames, ((0, 0), (lh, uh), (lw, uw), (0, 0)), mode='constant')
    return  pad_masks,pad_frames

def make_dataset(root):
    # lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root,'video')) if f.endswith('.npy')]
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'Image')) if f.endswith('.jpg')]
    img_list = [line.rstrip('\n') for line in lines]
    # img_list = [line.rstrip('\n') for line in lines]
    return [(os.path.join(root, 'Image', img_name + '.jpg'), os.path.join(root, 'Mask', img_name + '.png')) for img_name
            in img_list]


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    if len(np.shape(img)) == 3:#img
        img = Image.fromarray(img)
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)
    else:#mask
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
    return img

def im_to_torch_withoutNorm(img):
    if len(np.shape(img)) == 3:#img
        img = Image.fromarray(img)
        img = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(img)
    else:#mask
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
    return img

# def im_to_torch(img):
#     img = np.transpose(img, (2, 0, 1))  # C*H*W
#     img = to_torch(img.copy()).float()
#     return img

# def im_to_torch(img):
#     img = np.transpose(img, (2, 0, 1))  # C*H*W
#     img = to_torch(img.copy()).float()
#     return img

def in_get_subwindow_tracking(im, im_patch, param, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):
    top_pad, bottom_pad, left_pad, right_pad = param['pad']
    r, c, k = im.shape
    ori_r, ori_c, k = param['ori_size']
    context_xmin, context_xmax, context_ymin, context_ymax = param['context']

    if not np.array_equal(model_sz, original_sz):
        im_patch_original = cv2.resize(im_patch, (ori_c, ori_r))

    else:
        im_patch_original = im_patch
    im = np.zeros((r, c))

    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))
        # te_im = im_patch_original[top_pad:top_pad + r, left_pad:left_pad + c, :]
        te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)] = im_patch_original
        im = te_im[top_pad:top_pad + r, left_pad:left_pad + c]
    else:
        im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)] = im_patch_original

    return im


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    #     print(im.shape)
    # zzp: a more easy speed version
    if len(im.shape) == 3:
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = 0.0
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = 0.0
            if left_pad:
                te_im[:, 0:left_pad, :] = 0.0
            if right_pad:
                te_im[:, c + left_pad:, :] = 0.0
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1),
                                :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
    else:
        r, c = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c] = 0.0
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c] = 0.0
            if left_pad:
                te_im[:, 0:left_pad] = 0.0
            if right_pad:
                te_im[:, c + left_pad:] = 0.0
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
    #         im_patch = np.expand_dims(im_patch,-1)

    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    param = dict()
    param['pad'] = [top_pad, bottom_pad, left_pad, right_pad]
    param['context'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    param['ori_size'] = np.shape(im_patch_original)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch, param


def get_subwindow_trackin7g_search(im, pos, wh, original_sz, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    ih = np.shape(im)[0]
    iw = np.shape(im)[1]
    #     print(ih,iw)
    cx, cy = pos
    w, h = wh
    # print(cx,cy,w,h)
    a = np.random.randint(0, 20, 4)
    #     print(a)
    context_xmin = np.max((0, round(cx - 0.5 * w) - a[0]))
    context_xmax = np.min((iw, round(cx + 0.5 * w) + a[1]))
    context_ymin = np.max((0, round(cy - 0.5 * h) - a[2]))
    context_ymax = np.min((ih, round(cy + 0.5 * h) + a[3]))
    # print(context_xmax,context_xmin,context_ymax,context_ymin)
    # zzp: a more easy speed version
    if len(im.shape) == 3:
        r, c, k = im.shape
        #         print(im.shape)
        im_patch_original = im[int(context_ymin):int(context_ymax), int(context_xmin):int(context_xmax), :]

        # print(im_patch_original)
        im_patch = cv2.resize(im_patch_original, (original_sz, original_sz))

    else:
        r, c = im.shape
        #         if any([top_pad, bottom_pad, left_pad, right_pad]):
        #             te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad), np.uint8)
        #             te_im[top_pad:top_pad + r, left_pad:left_pad + c] = im
        #             if top_pad:
        #                 te_im[0:top_pad, left_pad:left_pad + c] = avg_chans
        #             if bottom_pad:
        #                 te_im[r + top_pad:, left_pad:left_pad + c] = avg_chans
        #             if left_pad:
        #                 te_im[:, 0:left_pad] = avg_chans
        #             if right_pad:
        #                 te_im[:, c + left_pad:] = avg_chans
        #             im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
        #         else:
        im_patch_original = im[int(context_ymin):int(context_ymax), int(context_xmin):int(context_xmax)]

        im_patch = cv2.resize(im_patch_original, (original_sz, original_sz))

    #         im_patch = np.expand_dims(im_patch,-1)

    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    param = dict()
    #     param['pad'] = [top_pad, bottom_pad, left_pad, right_pad]
    param['context'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    param['ori_size'] = np.shape(im_patch_original)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch, param
