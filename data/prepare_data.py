import json
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# def To_onehot(mask):
#     M = np.zeros((K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
#     for k in range(K):
#         M[k] = (mask == k).astype(np.uint8)
#     return M
#
#
# def All_to_onehot(masks):
#     Ms = np.zeros((K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
#     for n in range(masks.shape[0]):
#         Ms[:, n] = To_onehot(masks[n])
#     return Ms

def make_dataset_DUT_TR(root):
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'DUTS-TR-Image')) if
             f.endswith('.jpg') or f.endswith('.jpeg')]
    img_list = [line.rstrip('\n') for line in lines]
    return [(os.path.join(root, 'DUTS-TR-Image', img_name + '.jpg'), os.path.join(root, 'DUTS-TR-Mask', img_name + '.png'), img_name)
            for img_name in img_list]

def make_dataset_HKU_IS(root):
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'imgs')) if
             f.endswith('.png')]
    img_list = [line.rstrip('\n') for line in lines]
    return [(os.path.join(root, 'imgs', img_name + '.png'), os.path.join(root, 'gt', img_name + '.png'), img_name)
            for img_name in img_list]

def make_dataset_msra(root):
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root)) if
             f.endswith('.jpg') or f.endswith('.jpeg')]
    img_list = [line.rstrip('\n') for line in lines]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png'), img_name)
            for img_name in img_list]

def make_dataset1(root, root1, root2, video):
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, '480p', video)) if f.endswith('.jpg')]
    img_list = [line.rstrip('\n') for line in lines]
    img_list.sort(key=lambda x: x[5:-4])
    img_list.sort()
    images = []
    masks = []
    eyegazes = []
    for img_name in img_list:
        images.append(os.path.join(root, '480p', video, img_name + '.jpg'))
        masks.append(os.path.join(root1, '480p', video, img_name + '.png'))
        eyegazes.append(os.path.join(root2, video, img_name + '.png'))
    return images, masks, eyegazes, img_list


def make_dataset_davis(root, root1, video):
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, '480p', video)) if f.endswith('.jpg')]
    img_list = [line.rstrip('\n') for line in lines]
    img_list.sort(key=lambda x: x[5:-4])
    img_list.sort()
    images = []
    masks = []
    eyegazes = []
    for img_name in img_list:
        images.append(os.path.join(root, '480p', video, img_name + '.jpg'))
        masks.append(os.path.join(root1, '480p', video, img_name + '.png'))
    return images, masks, img_list

def make_dataset_davis2017(root, root1, video,dataname):
    if dataname == 'davis2017':
        root = os.path.join(root,'480p')
        root1 = os.path.join(root1,'480p')
    #返回所有的图片和掩码
    lines = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video)) if f.endswith('.jpg')]
    img_list = [line.rstrip('\n') for line in lines]
    img_list.sort(key=lambda x: x[5:-4])
    img_list.sort()
    images = []
    masks = []
    for img_name in img_list:
        images.append(os.path.join(root, video, img_name + '.jpg'))
        masks.append(os.path.join(root1, video, img_name + '.png'))
    return images, masks, img_list


def make_dataset_yvos(root, root1, video, obj_data):
    img_list= obj_data['frames']
    images = []
    masks = []
    for img_name in img_list:
        images.append(os.path.join(root, video, img_name + '.jpg'))
        masks.append(os.path.join(root1, video, img_name + '.png'))
    return images, masks, img_list

def mask_with_obj(masks,color):
    masks_individ = []
    images_individ = []
    img_list_individ = []
    for msk in masks:
        mask = Image.open(msk).convert('L')
        if color in [cl[-1] for cl in mask.getcolors()[1:]]: #颜色是否在图片内
            masks_individ.append(msk)
            images_individ.append(msk.replace('Annotations','JPEGImages').replace('png','jpg'))
            img_list_individ.append(msk.split('/')[-1][:-4])
    return img_list_individ, images_individ, masks_individ

out = []
for dataname in ['duts_quick']:

    if dataname == 'duts' or dataname == 'msra'or dataname == 'HKU-IS':
        if dataname == 'duts':
            root = '/home/ldz/文档/DUTS/DUTS-TR'
            image_dir = make_dataset_DUT_TR(root)
        if dataname == 'msra':
            root = '/home/ldz/文档/MSRA10K_Imgs_GT/Imgs'
            image_dir = make_dataset_msra(root)
        if dataname == 'HKU-IS':
            root = '/home/ldz/文档/HKU-IS'
            image_dir = make_dataset_HKU_IS(root)
        data = {}
        for img_dir, mask_dir, img_name in tqdm(image_dir):
            json_data = {'filename': dataname,
                         'image_dir': img_dir,
                         'image_name': img_name,
                         'mask_name': mask_dir,
                         'object_num': 1,
                         'length': 1
                         }
            out.append(json_data)

    if dataname == 'HKU-IS_quick':
        root = '/home/ldz/文档/HKU-IS'
        image_dir = make_dataset_HKU_IS(root)
        data = {}
        i = 0
        for img_dir, mask_dir, img_name in tqdm(image_dir):
            json_data = {'filename': 'HKU-IS',
                         'image_dir': img_dir,
                         'image_name': img_name,
                         'mask_name': mask_dir,
                         'object_num': 1,
                         'length': 1
                         }
            out.append(json_data)
            i += 1
            if i == 10:
                break

    if dataname == 'duts_quick':#如果不能过拟合，就说明模型有问题。
        root = '/home/ldz/文档/DUTS/DUTS-TR'
        image_dir = make_dataset_DUT_TR(root)
        data = {}
        i = 0
        for img_dir, mask_dir, img_name in image_dir:
            json_data = {'filename': 'duts',
                         'image_dir': img_dir,
                         'image_name': img_name,
                         'rmask_name': os.path.join('./data/mask/DUTS-TR/Mask2', img_name + '.png'),
                         'mask_name': mask_dir,
                         'object_num': 1,
                         'length': 1
                         }
            out.append(json_data)
            i += 1
            if i == 10:
                break

    if dataname == 'salicon':
        root = './data/eyegaze/train'
        image_dir = make_dataset_DUT_TR(root)
        data = {}
        for img_dir, mask_dir, img_name in image_dir:
            json_data = {'filename': 'salicon',
                         'image_dir': img_dir,
                         'image_name': img_name,
                         'mask_name': mask_dir,
                         'object_num': 1,
                         'length': 1
                         }
            out.append(json_data)

    if dataname == 'davis2016':
        root = '/home/ldz/文档/DAVIS/JPEGImages'
        root1 = '/home/ldz/文档/DAVIS/Annotations'
        lines = open('/home/ldz/文档/DAVIS/ImageSets/2016/train.txt', 'r')
        for line in lines:
            _video = line.rstrip('\n')
            images, masks, img_list = make_dataset_davis(root, root1, _video)
            image_num = len(img_list)
            mask = np.array(Image.open(masks[0]).convert('P'))
            shape = mask.shape
            # shape =
            # object_num = int(np.max(mask))  # for2017
            object_num = 1#for2016
            json_data = {'filename': 'davis2016',
                         'image_dir': images,
                         'image_name': img_list,
                         'mask_name': masks,
                         'object_num': object_num,
                         'length': image_num,
                         'shape' : shape
                         }
            out.append(json_data)
    if dataname == 'davis2016_val':
        root = '/home/ldz/文档/DAVIS/JPEGImages'
        root1 = '/home/ldz/文档/DAVIS/Annotations'
        lines = open('/home/ldz/文档/DAVIS/ImageSets/2016/val.txt', 'r')
        for line in lines:
            _video = line.rstrip('\n')
            images, masks, img_list = make_dataset_davis(root, root1, _video)
            image_num = len(img_list)
            mask = np.array(Image.open(masks[0]).convert('P'))
            shape = mask.shape
            # shape =
            # object_num = int(np.max(mask))  # for2017
            object_num = 1#for2016
            json_data = {'filename': 'davis2016_val',
                         'image_dir': images,
                         'image_name': img_list,
                         'mask_name': masks,
                         'object_num': object_num,
                         'length': image_num,
                         'shape' : shape
                         }
            out.append(json_data)
    if dataname == 'yvos':
        root = '/home/ldz/文档/Youtube_VOS/train/JPEGImages'
        root1 = '/home/ldz/文档/Youtube_VOS/train/Annotations'
        with open('/home/ldz/文档/Youtube_VOS/train/meta.json', 'r') as f:
            data = json.load(f)['videos']
            for key in data:
                masks_list = sorted([int(msk[:-4]) for msk in os.listdir(os.path.join(root1,key))])#数字排序
                masks_list = [str(msk).zfill(5) + '.png' for msk in masks_list]#变回“%05d.png”格式
                masks_list = [os.path.join(root1,key,msk) for msk in masks_list]
                # imgs_list = [msk.replace('png','jpg').replace('Annotations','JPEGImages') for msk in masks_list]
                mask_zero  = Image.open(masks_list[0]).convert("L")#第0帧mask，默认所有object在第一帧出场
                shape = mask_zero.size
                colors = [cl[-1] for cl in mask_zero.getcolors()[1:]]#获取颜色作为各个obj的索引号
                for color in colors:
                    img_list_individ, images_individ, masks_individ = mask_with_obj(masks_list, color)
                    json_data = {'filename': 'yvos',
                                 'color' : color,
                                 'image_dir': images_individ,
                                 'image_name': img_list_individ,
                                 'mask_name': masks_individ,
                                 'object_num': 1,
                                 'length': len(img_list_individ),
                                 'shape': shape
                                 }
                    out.append(json_data)

    if dataname == 'yvos_test':
        root = '/home/ldz/文档/Youtube_VOS/train/JPEGImages'
        root1 = '/home/ldz/文档/Youtube_VOS/train/Annotations'
        with open('/home/ldz/文档/Youtube_VOS/train/meta.json', 'r') as f:
            data = json.load(f)['videos']
            i =0
            for key in data:
                if i > 10:
                    break
                masks_list = sorted([int(msk[:-4]) for msk in os.listdir(os.path.join(root1,key))])#数字排序
                masks_list = [str(msk).zfill(5) + '.png' for msk in masks_list]#变回“%05d.png”格式
                masks_list = [os.path.join(root1,key,msk) for msk in masks_list]
                # imgs_list = [msk.replace('png','jpg').replace('Annotations','JPEGImages') for msk in masks_list]
                mask_zero  = Image.open(masks_list[0]).convert("L")#第0帧mask，默认所有object在第一帧出场
                shape = mask_zero.size
                colors = [cl[-1] for cl in mask_zero.getcolors()[1:]]#获取颜色作为各个obj的索引号
                for color in colors:
                    img_list_individ, images_individ, masks_individ = mask_with_obj(masks_list, color)
                    json_data = {'filename': 'yvos',
                                 'color' : color,
                                 'image_dir': images_individ,
                                 'image_name': img_list_individ,
                                 'mask_name': masks_individ,
                                 'object_num': 1,
                                 'length': len(img_list_individ),
                                 'shape': shape
                                 }
                    out.append(json_data)
                i += 1



    if dataname == 'davis2017' or dataname == 'tianchiyusai':
        if dataname == 'davis2017':
            root = '/home/ldz/文档/DAVIS/JPEGImages'
            root1 = '/home/ldz/文档/DAVIS/Annotations'
            lines = open('/home/ldz/文档/DAVIS/ImageSets/2017/train.txt', 'r')
        elif dataname == 'tianchiyusai':
            root = '/home/ldz/文档/tianchiyusai/JPEGImages'
            root1 = '/home/ldz/文档/tianchiyusai/Annotations'
            lines = open('/home/ldz/文档/tianchiyusai/ImageSets/train.txt', 'r')
        for line in tqdm(list(lines)):
            _video = line.rstrip('\n')
            images, masks, img_list = make_dataset_davis2017(root, root1, _video, dataname)
            image_num = len(img_list)
            mask = Image.open(masks[0]).convert('L')
            colors = [cl[-1] for cl in mask.getcolors()[1:]]
            mask = np.array(mask)
            shape = mask.shape

            object_num = len(np.unique(mask))-1 #去掉背景
            # if object_num ==1 :
            #     continue

            for color in colors:
                img_list_individ,images_individ, masks_individ = mask_with_obj(masks,color)

                json_data = {'filename': dataname,
                             'color' : color,
                             'image_dir': images_individ,
                             'image_name': img_list_individ,
                             'mask_name': masks_individ,
                             'object_num': 1,
                             'length': len(img_list_individ),
                             'shape': shape
                             }
                out.append(json_data)



    with open('./data_{}.json'.format(dataname), 'w') as f:
        json.dump(out, f,ensure_ascii=False)
    print("success for {}!".format(dataname))
