import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import transforms
# from config import msra10k_path
from datasets_loader import ImageFolder
from datasets_loader import DAVIS_MO_Test
from utils.misc import AvgMeter, check_mkdir
# from model_efnet_v4 import SiamET
from models.model_EMN_SelectMem import EMN
from models.model_STM import STM
from torch.backends import cudnn
from utils import joint_transforms
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
from train_utils import iou
from train_utils import toCudaVariable
from train_utils import train_3f1o

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

# Initialize Horovod
hvd.init()
# Horovod Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

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
                        default=3,type=int)
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


cudnn.benchmark = True

torch.manual_seed(2018)
# torch.cuda.set_device(3)
root = './data/mask/DUTS-TR'
if hvd.rank() == 0:
    writer = SummaryWriter('runs/pre_EMN{}'.format(time.strftime("%m%d%H%M%S", time.localtime())))

ckpt_path = './ckpt'
exp_name = 'mask_9'
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
msra10k_path = root

# args = {
#     'iter_num': 10,
#     ##todo:以下为修改版
#     'train_batch_size': 2,
#     ##todo:以下为原版
#     # 'train_batch_size': 10,
#     'last_iter': 0,
#     'lr': 1e-5,
#     'lr_decay': 0.5,
#     'weight_decay': 5e-6,
#     'momentum': 0.9,
#     'snapshot': '120',
#     'loadepoch':'-1'
# }

target_transform = transforms.ToTensor()
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])

duts_set = ImageFolder('./data/data_duts.json', joint_transform=joint_transform, target_transform=target_transform)
msra_set = ImageFolder('./data/data_msra.json', joint_transform=joint_transform, target_transform=target_transform)
HKU_IS_set = ImageFolder('./data/data_HKU-IS.json', joint_transform=joint_transform, target_transform=target_transform)
train_set = ConcatDataset([duts_set]+[msra_set]+[HKU_IS_set])
# train_set = ImageFolder('./data/data_HKU-IS_quick.json', joint_transform=joint_transform, target_transform=target_transform)
# train_set = ImageFolder('./data/data_duts_quick.json', joint_transform=joint_transform, target_transform=target_transform)

# Horovod Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set, num_replicas=hvd.size(), rank=hvd.rank())

# train_loader = DataLoader(train_set, batch_size=arg_load.batchsize, num_workers=0, shuffle=True)#无horovod，bz=1
train_loader = DataLoader(train_set, batch_size=arg_load.batchsize, sampler=train_sampler)

DATA_ROOT = '/home/ldz/文档/DAVIS'
Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(16,'val'), single_object=True)
Testloader = DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

# ##todo:以下修正版
# criterion1 = nn.BCELoss().cuda()
#todo:以下原版
criterion1 = nn.CrossEntropyLoss().cuda()  # nn.BCEWithLogitsLoss().cuda()




def main():
    net = EMN()
    # net = STM()
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=arg_load.lr,weight_decay=arg_load.weight_decay)

    if arg_load.loadepoch != '-1':
        print('Loading checkpoint @Epoch {}...'.format(arg_load.loadepoch))
        load_name = os.path.join(ckpt_path,exp_name,'{}.pth'.format(arg_load.loadepoch))
        checkpoint = torch.load(load_name)

        # 相当于用''代替'module.'。
        # 直接使得需要的键名等于期望的键名。
        net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        # net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

        del checkpoint
        torch.cuda.empty_cache()  # 不清理有可能报error:137

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

        # # if idx % 3 == 0 and idx != 0:
        # if 0 == 0:
        #     evalution(net, Testloader,writer,idx)

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = arg_load.lr * (1 - float(curr_iter) / arg_load.iter_num) ** arg_load.lr_decay
            # optimizer.param_groups[0]['lr'] = arg_load.lr * (arg_load.lr_decay ** (curr_iter))
            #todo：以下为修改版
            img_ref, mask_ref,img_mid, mask_mid, img_target, mask_target = data
            
            mask_ref[mask_ref>0.5]=1
            mask_ref[mask_ref<=0.5]=0

            mask_mid[mask_mid > 0.5] = 1
            mask_mid[mask_mid <= 0.5] = 0

            mask_target[mask_target > 0.5] = 1
            mask_target[mask_target <= 0.5] = 0
            
            batch_size =img_ref.size(0)
            img_ref, mask_ref,img_mid, mask_mid, img_target, mask_target = toCudaVariable([img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target])
            Fs = torch.cat([img.unsqueeze(2) for img in [img_ref,img_mid,img_target]],dim=2)
            Ms_obj = torch.cat([mask_ref,mask_mid,mask_target],dim=1).unsqueeze(1)
            Ms_None = torch.zeros_like(Ms_obj).expand(-1,9,-1,-1,-1)
            Ms = torch.cat([1 - Ms_obj, Ms_obj, Ms_None], dim=1)

            img_target_temp = img_target.detach().cpu()
            mask_target_temp = mask_target.detach().cpu()
            del img_ref, mask_ref, img_mid, mask_mid, img_target, mask_target
            torch.cuda.empty_cache()  # 不清理有可能报error:137

            loss0, Es, pred = train_3f1o(model=net,
                                   optimizer=optimizer,
                                   loss_func=criterion1,
                                   Fs=Fs,
                                   Ms=Ms,
                                   num_frames=3,
                                   num_objects=1,
                                   Mem_every=1,
                                   Mem_number=None)
            total_loss = loss0
            total_loss_record.update(total_loss.item(), batch_size)
            loss0_record.update(loss0.item(), batch_size)
            loss1_record.update(loss0.item(), batch_size)
            loss2_record.update(loss0.item(), batch_size)
            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [lossr %.5f], [lr %.13f][iou %.5f]' % \
                  (idx, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   optimizer.param_groups[0]['lr'],iou(Es[:,1,2],Ms[:,1,2]))
            print(log)

            global_step = idx * len(train_loader) + i

            if i % 50 == 0 and hvd.rank() == 0:
                print('add point to Tensorboard')
                writer.add_scalar('training loss', float(total_loss), global_step)
                maskedimg = transforms.ToPILImage()(img_target_temp[0, :, :, :]).convert('RGB')
                maskedimg.save('pre_img_target.jpg')
                eyeout = transforms.ToPILImage()(mask_target_temp[0, :, :, :]).convert('L')
                eyeout.save(os.path.join('pre_mask_target_gt.png'))
                output = transforms.ToPILImage()(Es.data.cpu()[0, 1, 2]).convert('L')
                output.save(os.path.join('pre_output1.png'))

                writer.add_image('maskedimg', transforms.ToTensor()(maskedimg))
                writer.add_image('main_mask_target_gt', transforms.ToTensor()(eyeout))
                writer.add_image('main_output1', transforms.ToTensor()(output))
                writer.close()

            # Then we save models by steps [only 'rank==0' can run]
            if (global_step+1) % 200 == 0 and hvd.rank() == 0 :  # 避免测试时覆盖权重的问题
            # if  hvd.rank() == 0 :  # 避免测试时覆盖权重的问题
                save_state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'idx': idx}
                torch.save(
                    obj=save_state,
                    f=os.path.join(ckpt_path, exp_name, 'pre_EMN_Resnest101_step_latest.pth')
                )
                print('After {} steps, we saved a step model.'.format(global_step))

        # Then save two models [only 'rank==0' can run]
        if idx % 1 == 0 and hvd.rank() == 0 :  # 避免测试时的覆盖权重问题
            save_state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'idx': idx}
            torch.save(
                obj=save_state,
                f=os.path.join(ckpt_path, exp_name, 'pre_EMN_Resnest101_{}.pth'.format(idx))
            )
            torch.save(
                obj=save_state,
                f=os.path.join(ckpt_path, exp_name, 'pre_EMN_Resnest101_latest.pth')
            )
            print('{}th idx ended, we saved 2 models.'.format(idx))

        curr_iter += 1

if __name__ == '__main__':
    main()
