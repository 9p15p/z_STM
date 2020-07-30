#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/19 下午9:41
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : overlay.py

import argparse

from PIL import Image
import matplotlib.pyplot as plt
import os
def if_not_exist_then_creat(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def overlay_one_kind(msk_path, img_path, ovly_dir):
    msks = os.listdir(msk_path)
    for i,_msk_ in enumerate(msks):

        img = os.path.join(img_path,_msk_[:-4]+'.jpg')
        msk = os.path.join(msk_path,_msk_)

        image = Image.open(img).convert("RGBA")
        mask  = Image.open(msk).convert("RGBA")
        image_ovly = Image.blend(image,mask,0.6).convert("RGB")
        image_ovly_name = os.path.join(ovly_dir,_msk_[:-4]+'.jpg')
        image_ovly.save(image_ovly_name)

def get_arguments():
    parser = argparse.ArgumentParser(description="overlay")
    parser.add_argument("-year", type=int, help="year(16 or 17)",default='17')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()

    DAVIS_JPEG_PATH = '/home/ldz/文档/DAVIS/JPEGImages/480p'
    INFER_MASK_PATH = os.path.join('/home/ldz/temp_project/z_STM/test', 'STM_DAVIS_' + str(args.year) + 'val')

    masks = os.listdir(INFER_MASK_PATH)
    print("we overlay 20{}'s test".format(args.year))
    for i,_msk_path_ in enumerate(masks):
        img_path = os.path.join(DAVIS_JPEG_PATH, _msk_path_)
        msk_path = os.path.join(INFER_MASK_PATH, _msk_path_)

        ovly_dir = os.path.join('/home/ldz/temp_project/z_STM/test/overlay20' + str(args.year),_msk_path_)
        if_not_exist_then_creat(ovly_dir)

        overlay_one_kind(msk_path, img_path, ovly_dir)

        print(_msk_path_+" finished!")

    for root, dirs, files in os.walk(f"/home/ldz/temp_project/z_STM/test/overlay20{args.year}", topdown=False):
        os.system('ffmpeg -f image2 -i {}/%05d.jpg  -vcodec libx264 -r 10  {}/test.mp4 -y'
                  .format(root, root))

    print("**************")
    print("Congratulations, all finished!")

