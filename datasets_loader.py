import glob
import json
import os.path
import random

import torch.utils.data as data

from dataset.dataset_utils import *
from utils.joint_transforms import RandomRotate
from train_utils import size_640_384

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root = None, joint_transform=None, transform=None, target_transform=None):
        with open(root) as f :
            self.file = json.load(f)

        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.nums = [x for x in range(6)]
        self.size_t = 127
        self.size_s = 255
        self.frame_range = 100

    def get_box(self, label):
            label = label.copy()
            thre = 0  # np.max(label)*0.5
            label[label > thre] = 1
            label[label <= thre] = 0
            a = np.where(label != 0) #找到符合condition要求的位置，并产生一个tuple,tuple[0]为x坐标，tuple[y]为y坐标
            h,w = np.shape(label)[:2]
            if len(a[0]) != 0:
                bbox1 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            else:
                bbox1 = [0, h, 0, w]
            # bbox1 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

            x1 = int(bbox1[2])
            y1 = int(bbox1[0])
            w1 = int(bbox1[3] - bbox1[2])
            h1 = int(bbox1[1] - bbox1[0])
            target_pos = np.array([x1 + w1 / 2, y1 + h1 / 2])#position
            target_sz = np.array([w1, h1])#size
            if w1==0 and h1!=0:
                target_sz = np.array([h1, h1])
            if w1!=0 and h1==0:
                target_sz = np.array([w1, w1])
            if w1==0 and h1==0:
                target_sz = np.array([w, h])
            return target_pos, target_sz

    def get_temp(self, im, target_sz, target_pos, exp_size=127, opsize=0.0):
            avg_chans = np.mean(im, axis=(0, 1))

            wc_z = target_sz[0] + opsize * sum(target_sz)
            hc_z = target_sz[1] + opsize * sum(target_sz)
            s_z = round(np.sqrt(wc_z * hc_z))
            # initialize the exemplar
            z_crop, param = get_subwindow_tracking(im, target_pos, exp_size, s_z, avg_chans)
            return z_crop

    def get_search(self, im, target_sz, target_pos, exp_size=255, opsize=0.0):
            avg_chans = np.mean(im, axis=(0, 1))

            wc_x = target_sz[0] + opsize * sum(target_sz)
            hc_x = target_sz[1] + opsize * sum(target_sz)
            s_x = round(np.sqrt(wc_x * hc_x))
            x_crop, param = get_subwindow_tracking(im, target_pos, exp_size, s_x, avg_chans)
            return x_crop

    def get_crop(self, im, target_sz, target_pos, exp_size=255, opsize=0.0):
            h = im.shape[0]
            w = im.shape[1]
            wc_x = target_sz[0] + opsize * target_sz[0]
            hc_x = target_sz[1] + opsize * target_sz[1]
            crop_x1 = int(max(0, round(target_pos[0] - wc_x * 0.5)))
            crop_y1 = int(max(0, round(target_pos[1] - hc_x * 0.5)))
            crop_x2 = int(min(w, round(target_pos[0] + wc_x * 0.5)))
            crop_y2 = int(min(h, round(target_pos[1] + hc_x * 0.5)))

            if len(im.shape) == 3:
                out = im[crop_y1:crop_y2, crop_x1:crop_x2, :]
            else:
                out = im[crop_y1:crop_y2, crop_x1:crop_x2]
            out = cv2.resize(out, (exp_size, exp_size))
            return im_to_torch(out)

    def get_file_idx(self, file_path):
            files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
            files.sort(key=lambda x: x[5:-4])
            files.sort()
            start = np.random.randint(1, len(files))
            pro_files = [files[0], files[start]]
            return pro_files
    def get_move(self, target_size,target_pos):
        wrange = int(min(0.3*target_size[0],0.3*target_size[1]))
        if random.random() <0.5:
            random_w = -1 * wrange
            random_h = -1 * wrange
        else:
            random_w =  wrange
            random_h =  wrange
        new_target_pos = target_pos + [random_w,random_h]
        return new_target_pos
    def __getitem__(self, index):
        train_data = self.file[index]
        data_source = train_data.get('filename')

        if data_source == 'duts' or data_source == 'msra'or data_source =='HKU-IS':
            # PRE_TRAIN_SHAPE = (384,384)
            PRE_TRAIN_SHAPE = size_640_384
            img_path = train_data.get('image_dir')
            mask_path = train_data.get('mask_name')
            # rmask_path = train_data.get('rmask_name')
            img = Image.open(os.path.join(img_path)).convert('RGB')
            mask = Image.open(os.path.join(mask_path)).convert('L')
            # rmask = Image.open(os.path.join(rmask_path)).convert('L')

            #todo：以下为修改版
            img_ref = img.resize(PRE_TRAIN_SHAPE)
            mask_ref = mask.resize(PRE_TRAIN_SHAPE)
            img_mid, mask_mid = self.joint_transform(img_ref, mask_ref)
            img_target, mask_target = RandomRotate(20)(img_mid,mask_mid)#img无用，只是为了满足格式
            # # Normalization在model的encoder中实现
            img_ref = im_to_torch_withoutNorm(np.array(img_ref))
            mask_ref = im_to_torch_withoutNorm(np.array(mask_ref))
            img_mid = im_to_torch_withoutNorm(np.array(img_mid))
            mask_mid = im_to_torch_withoutNorm(np.array(mask_mid))
            img_target = im_to_torch_withoutNorm(np.array(img_target))
            mask_target = im_to_torch_withoutNorm(np.array(mask_target))

            return img_ref, mask_ref,img_mid, mask_mid, img_target, mask_target

        if data_source == 'davis2017' or data_source == 'yvos'or data_source == 'tianchiyusai':

            color = int(train_data.get('color'))
            img_len = 3
            imgs_path = train_data.get('image_dir')

            while len(imgs_path) < 3 :  #防止sample的时候长度不够。
                imgs_path.append(imgs_path[-1])

            imgs_path = sorted(random.sample(imgs_path, img_len),key= lambda x:int(x[-9:-4]))

            masks_path = [img_path.replace('JPEGImages','Annotations').replace('jpg','png') for img_path in imgs_path]
            origin_shape = train_data.get('shape')

            return color, imgs_path, masks_path, origin_shape, img_len

        if data_source == 'davis2016':
            img_num = train_data.get('length')
            obj_num = train_data.get('object_num')
            img_paths = train_data.get('image_dir')
            mask_paths = train_data.get('mask_name')
            origin_shape = train_data.get('shape')
            shape = (384, 384)
            # if img_paths[0].split('/')[-2] != 'bmx-bumps':
            #     inf = {}
            #     inf['num_obj'] = obj_num
            #     return 1,1,inf
            # todo:以下为修改版
            all_F = torch.empty((img_num,) + (3,) + shape)
            all_M = torch.empty((obj_num,) + (img_num,) + (1,) + shape)
            inf = {}
            for i, img_path in enumerate(img_paths):
                all_F[i] = im_to_torch_withoutNorm(np.array(Image.open(img_path).resize(shape).convert('RGB')))
            for i, mask_path in enumerate(mask_paths):
                masks = im_to_torch_withoutNorm(np.array(Image.open(mask_path).resize(shape).convert('L'), dtype=np.uint8))
                all_M[0,i] = torch.where(masks !=0 , torch.ones_like(masks), torch.zeros_like(masks)).to(torch.uint8)


            inf['name'] = img_path.split('/')[-2]
            inf['num_frames'] = img_num
            inf['num_obj'] = obj_num
            inf['origin_shape'] = origin_shape
            return all_F.permute((1, 0, 2, 3)), all_M.permute((2, 0, 1, 3, 4))[0], inf


    def __len__(self):
        return len(self.file)

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                # _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P").resize((384, 384)))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P")
                self.size_480p[_video] = np.shape(_mask480)
                # _mask480 = np.array(_mask480.resize((384, 384)))
                _mask480 = np.array(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            # N_frames[f] = np.array(Image.open(img_file).convert('RGB').resize((384,384))) / 255.
            N_frames[f] = np.array(Image.open(img_file).convert('RGB')) / 255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
                # N_masks[f] = np.array(Image.open(mask_file).convert('P').resize((384,384)), dtype=np.uint8)
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                N_masks[f] = 255

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info
