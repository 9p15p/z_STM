from __future__ import division

import argparse

# torch
import torch
import torch.nn.functional as F


# general libs


def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='RGMP')
    parser.add_argument('--epochs', dest='num_epochs',
                        help='number of epochs to train',
                        default=2_000_000, type=int)
    parser.add_argument('--bs', dest='bs',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='display interval',
                        default=1, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                        help='interval of epochs to perform validation',
                        default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default='saved_models', type=str)

    # BPTT
    parser.add_argument('--bptt', dest='bptt_len',
                        help='length of BPLoading weights:TT',
                        default=12, type=int)
    parser.add_argument('--bptt_step', dest='bptt_step',
                        help='step of truncated BPTT',
                        default=4, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-7, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=3, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                        help='epoch to load model',
                        default='-1', type=str)

    args = parser.parse_args()
    return args

class font:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def ToLabel(E):
   E=(E > torch.ones_like(E)*0.5).to(torch.uint8)#sth over threshold get "1"
   return E

def iou(pred, gt):
   pred = pred.squeeze()
   pred = ToLabel(pred)
   gt = gt.squeeze()
   agg = pred + gt
   i = float(torch.sum(agg == 2))
   u = float(torch.sum(agg > 0))
   # if i/u == 0.0:
   #     print("no")
   try:
       return i / u
   except:
       return 0


