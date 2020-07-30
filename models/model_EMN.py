from __future__ import division
# general libs
from __future__ import division
import torch.nn as nn
from torchvision import models
from resnest.torch import resnest101

# general libs
import math
from utils.helpers import *
from models.ASPP import ASPP
from models.DepthCorr import DepthCorr

# get list of models

print('Space-time Memory Networks: initialized.')


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        # resnet = resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True);resnest_name='resnest50'
        # resnet = resnest101(pretrained=True);resnest_name='resnest101'  # marked

        if resnest_name == 'resnest101':
            self.conv1_m = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        # resnet = resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        # resnet = resnest101(pretrained=True)  #marked

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        # m4 = self.ResMM(r4)
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

        return p  # , p2, p3, p4


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H * W)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW


        mo = m_out.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p# mem_out是通过相似度函数找出来的value和Query value的结合体，P是两个key的相似度函数


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)

class EMN(nn.Module):
    def __init__(self):
        super(EMN, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)
        # self.KV_Q_r4_new = KeyValue(2048, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
        self.ASPP = ASPP(in_channels=1024, atrous_rates=(6, 12, 18))
        # self.DepthCorr = DepthCorr(in_channels=1024, hidden=2048, out_channels=1024)

    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            batch_size = mem.shape[0]
            pad_mem = ToCuda(torch.zeros(batch_size, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[:, 1, :, 0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects):
        # memorize a frame
        num_objects = num_objects[0].item()
        B, K, H, W = masks.shape  # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f': [], 'm': [], 'o': []}
        for o in range(1, num_objects + 1):  # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:, o])
            B_list['o'].append((torch.sum(masks[:, 1:o], dim=1) + \
                                torch.sum(masks[:, o + 1:num_objects + 1], dim=1)).clamp(0, 1))

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4)  # batch_size, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        batch_size, H, W = ps.shape
        em = ToCuda(torch.zeros(batch_size, K, H, W))
        # em[:, 0] = torch.prod(1 - ps, dim=0)  # bg prob 背景就是所有层里没有object的位置
        em[:, 0] = 1 - ps  # bg prob 背景就是所有层里没有object的位置
        # em[:, 1:batch_size + 1] = ps  # obj prob
        em[:, 1] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    # def conv2d_dw_group(self, x, temp, pad=False):
    #     """ A batch-wise 2d image correlation in terms of vectors in channel dimension.
    #     from https://github.com/foolwood/SiamMask/blob/0eaac33050fdcda81c9a25aa307fffa74c182e36/models/rpn.py#L32
    #     @ Args:
    #         x: input tensor with shape (b, C, H, W)
    #         temp: template with shape (b, C, h, w)
    #         pad: if pad, outH == H and outW == W
    #     @ Returns:
    #         out: output feature with shape (b, C, outH, outW)
    #     """
    #     b, C, H, W = x.shape
    #     x = x.view(1, b * C, x.size(2), x.size(3))  # 1, b*c, H, W
    #     temp = temp.view(b * C, 1, temp.size(2), temp.size(3))  # b*c, 1, h, w
    #
    #     if pad:
    #         _, _, h, w = temp.shape
    #         out = F.conv2d(x, temp, groups=b * C, padding=(h // 2, w // 2))
    #         out = F.interpolate(out, size=(H, W))
    #         # out = out[:, :, :H, :W]  # 1, b*C, H, W
    #     else:
    #         out = F.conv2d(x, temp, groups=b * C)  # 1, b*C, h', w'
    #
    #     out = out.view(b, C, out.size(2), out.size(3))
    #     return out

    def segment(self, frame, keys, values, num_objects):
        num_objects = num_objects[0].item()
        batch_size, K, keydim, T, H, W = keys.shape  # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)

        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16

        # # todo:siamese encoder_q begin
        # r4_, _, _, _, _ = self.Encoder_Q(imsk_1st_qtr)
        #
        # # r4_new = self.conv2d_dw_group(r4,r4_,pad=True)
        # r4_new = self.DepthCorr(r4_, r4)
        # c_feat = torch.cat((r4, r4_new), dim=1)  # C == 2048# along channel dim.
        # # c_new = nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=3,stride=1,padding=(1,1)).cuda()(c_feat)
        # k4, v4 = self.KV_Q_r4_new(c_feat)  # 1, dim, H/16, W/16
        # # todo:siamese encoder_q end

        # expand to ---  bz, c, h, w
        k4e, v4e = k4.expand(batch_size, -1, -1, -1), v4.expand(batch_size, -1, -1, -1)
        r3e, r2e = r3.expand(batch_size, -1, -1, -1), r2.expand(batch_size, -1, -1, -1)

        # memory select kv:(bz, K, C, T, H, W)
        m4, viz= self.Memory(keys[:, 1], values[:, 1], k4e, v4e)
        m4_new = self.ASPP(m4)
        logits = self.Decoder(m4_new, r3e, r2e)  # 有skip connection
        ps = F.softmax(logits, dim=1)[:, 1]  # no, h, w
        # ps = indipendant possibility to belong to each object

        logit = self.Soft_aggregation(ps, K)  # bz, K, H, W
        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        return logit

    def forward(self, *args, **kwargs):
        # print(args[2].dim() > 4)
        if args[2].dim() > 4:  # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)



if __name__ == '__main__':
    pass
