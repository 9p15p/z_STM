#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 17:56
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn

#删除一些冗余的节点文件。

import os
for root, dirs, files in os.walk("../ckpt/mask_9", topdown=False):
    for file in files:
        if '_STM_' in file:
            file_path = os.path.join(root,file)
            os.remove(file_path)
            print('delete' + file_path)