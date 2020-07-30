from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
from datasets_loader import DAVIS_MO_Test
from models.model_EMN_SelectMem import EMN as STM


torch.set_grad_enabled(False) # Volatile
time_log = []
def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-mem", type=int, help="Mem_every", default=5)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    parser.add_argument("-loadepoch", type=str, help="loadepoch", default='main_STM_latest.pth')
    return parser.parse_args()

args = get_arguments()

GPU = args.g
YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on DAVIS')
print("torch.cuda.is_available():",torch.cuda.is_available())
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

def pad_divide_by(in_list, d, in_size):
    """pad and resize pics(in 'in_list') to those can be exact divided by 'd'

    :param in_list:
    :param d:
    :param in_size:
    :return:out_list , pad_array
    """
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array

def crop_obj(mask_first, img_first, obj_msk_shape):
    try:
        h_min = torch.min(torch.where(mask_first)[0])
        h_max = torch.max(torch.where(mask_first)[0])
        w_min = torch.min(torch.where(mask_first)[1])
        w_max = torch.max(torch.where(mask_first)[1])
    except:
        h_min = 0
        h_max = mask_first.shape[0]
        w_min = 0
        w_max = mask_first.shape[1]


    img_obj = (mask_first * img_first)[0][:,h_min:h_max+1,w_min:w_max+1]
    img_obj = transforms.ToTensor()(transforms.ToPILImage()(img_obj.detach().cpu()).resize(obj_msk_shape)).unsqueeze(0)
    return img_obj.cuda()

def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    time_Run_video_begin = time.time()
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    # msk_shape = (Ms[0, 1, 0].size()[0], Ms[0, 1, 0].size()[1])
    # [img_first, mask_first], pad = pad_divide_by([Fs[:, :, 0], Ms[0, 1, 0]], 16, msk_shape)
    # imsk_1st_qtr = transforms.ToTensor()(transforms.ToPILImage()((mask_first * img_first)[0].detach().cpu()).resize((int(msk_shape[1] / 4), int(msk_shape[0] / 4)))).unsqueeze(0) #Image的顺序是W,H
    time1 = time.time()
    msk_shape = (Ms[0, 1, 0].size()[0], Ms[0, 1, 0].size()[1])
    obj_msk_shape = (int(msk_shape[1] / 4), int(msk_shape[0] / 4))  # Image先宽后高
    [img_first, mask_first], pad = pad_divide_by([Fs[:, :, 0], Ms[0, 1, 0]], 16, msk_shape)
    imsk_1st_qtr = crop_obj(mask_first, img_first, obj_msk_shape)
    time2 = time.time()
    sentence = "crop_obj cost_time:{} s".format(time2-time1)
    #print(sentence)
    time_log.append(sentence)

    for t in tqdm.tqdm(range(1, num_frames)):

        time1 = time.time()

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects]))
        time2 = time.time()
        sentence = "Memorize cost_time:{} s".format(time2 - time1)
        #print(sentence)
        time_log.append(sentence)

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            # this_keys = (t - 1) / t * keys + 1/t * prev_key
            # this_values = (t - 1) / t * values + 1/t * prev_value
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        time1 = time.time()

        # segment
        with torch.no_grad():
            logit = model(Fs[:, :, t], imsk_1st_qtr, this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        time2 = time.time()
        sentence = "Segmentation cost_time:{} s".format(time2 - time1)
        #print(sentence)
        time_log.append(sentence)

        # todo:使用上一帧作为参考begin
        output = Es[:,:,t]
        [img_last, mask_last], pad = pad_divide_by([Fs[:, :, t], output[0, 1]], 16, msk_shape)
        imsk_1st_qtr = crop_obj(mask_last, img_last, obj_msk_shape)
        # todo:使用上一帧作为参考end

        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values

        del logit, prev_key, prev_value, this_keys, this_values
        torch.cuda.empty_cache()  # 不清理有可能报error:137

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    time_Run_video_end = time.time()
    sentence = "total_Run_video_function_for_once cost_time:{} s".format(time_Run_video_end - time_Run_video_begin)
    #print(sentence)
    time_log.append(sentence)
    return pred, Es




Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

model = nn.DataParallel(STM())
# model = STM()
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN


# pth_path = '/home/ldz/temp_project/z_STM/ckpt/mask_9/main_STM2016_99.pth'
pth_path = os.path.join('/home/ldz/temp_project/z_STM/ckpt/mask_9/',args.loadepoch+'.pth')
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

# # new_state_dict = {k.replace('module.',''):v for k,v in torch.load(pth_path).items()}
# new_state_dict = {'module.'+k:v for k,v in torch.load(pth_path).items()}
# model.load_state_dict(
#     new_state_dict
# )
# del new_state_dict
# torch.cuda.empty_cache() #清空显存，没有可能会报错137

code_name = '{}_DAVIS_{}{}'.format(MODEL,YEAR,SET)
print('Start Testing:', code_name)



for seq, V in enumerate(Testloader):

    # #resize 384*384版本
    # Fs_, Ms_, num_objects, info = V
    # B, C, T, H, W = Fs_.shape
    # Fs = torch.zeros((B, C, T, 384, 384))
    # B, C, T, H, W = Ms_.shape
    # Ms = torch.zeros((B, C, T, 384, 384))
    # origin_resize = (int(info['size_480p'][1]),int(info['size_480p'][0])) #Image的顺序是W,H
    # for i in range(T):
    #     Fs[0, :, i] = transforms.ToTensor()(transforms.ToPILImage()(Fs_[0, :, i]).resize((384, 384)))
    #     for j in range(C):
    #         Ms[0, j, i] = transforms.ToTensor()(transforms.ToPILImage()(Ms_[0, j, i]).resize((384, 384)))
    # seq_name = info['name'][0]
    # num_frames = info['num_frames'][0].item()
    # num_objects = int(num_objects[0][0])
    # pred_list = np.zeros((num_objects + 1, T, 384, 384), dtype=np.uint8)  # 第一层是背景用来记录为0

    # 无resize版本
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    num_objects = int(num_objects[0][0])
    origin_resize = (int(info['size_480p'][1]), int(info['size_480p'][0]))  # Image的顺序是W,H
    B, C, T, H, W = Fs.shape
    pred_list = np.zeros((num_objects + 1, T, H, W), dtype=np.uint8)  # 第一层是背景用来记录为0


    Fs = Fs.cuda()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames,num_objects))
    for i in range(num_objects):
        Ms_new = torch.zeros_like(Ms)
        Ms_new[0,1,0] = Ms[0,i+1,0] #Ms_new实际上只用到了[0,0,0]和[0,1,0]
        Ms_new = Ms_new.cuda()
        pred, Es = Run_video(Fs, Ms_new, num_frames, num_objects=1, Mem_every=args.mem, Mem_number=None)
        pred_list[i+1] = pred
        del Ms_new
        torch.cuda.empty_cache()  # 不清理有可能报error:137


    time1 = time.time()
    pred = np.argmax(pred_list, axis=0).astype(np.uint8)
    time2 = time.time()
    sentence = "serialization of Mask cost_time:{} s".format(time2 - time1)
    #print(sentence)
    time_log.append(sentence)

    time1 = time.time()
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        # img_E = Image.fromarray(pred[f])#.resize(origin_resize)
        img_E = Image.fromarray(pred[f]) .resize(origin_resize)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
    time2 = time.time()
    sentence = "save_the_result_of_a_video cost_time:{} s".format(time2 - time1)
    #print(sentence)
    time_log.append(sentence)

    fh = open('time_log_{}.txt'.format(time.strftime("%m%d%H%M", time.localtime())),
              mode='w',
              encoding='utf-8')
    for sen in time_log:
        fh.write(sen)
        fh.write('\n')
    fh.close()
    print("Save time_log successfully!")


    if VIZ:
        from utils.helpers import overlay_davis
        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        for f in range(num_frames):
            pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
print("Congratulations, we finished!")



