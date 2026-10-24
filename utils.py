import torch
from torch.utils.data import Dataset
import numpy as np
import logging 
import time 
import os
import cv2
import math 
import albumentations
from sklearn.metrics import mean_squared_error as MSE
import torch.nn as nn

class TrainData(Dataset):
    def __init__(self,args):
        self.img_path = args.train_data_path
        self.size_h, self.size_w = args.size
        num_patch = 50
        img_names = os.listdir(self.img_path)
        # print('img_names:', img_names)
        self.img_names = img_names * num_patch

    def __getitem__(self,index):
        gt = np.zeros([self.size_h, self.size_w],dtype=np.float32)

        image = cv2.imread(os.path.join(self.img_path,self.img_names[index]))
        image_h, image_w = image.shape[:2]
        if image_h > image_w:
            image = cv2.flip(image, 1)
            image = cv2.transpose(image)
            image_h, image_w = image.shape[:2]

        crop_flag = np.random.randint(0,10)>5
        crop_h = np.random.randint(self.size_h//2,image_h)
        if crop_flag:
            crop_w = np.random.randint(self.size_w//2,image_w)
        else:
            crop_w = crop_h     

        transform = albumentations.Compose([
            albumentations.RandomCrop(height=crop_h,width=crop_w,p=1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Resize(self.size_h,self.size_w)
        ])
        gt = transform(image=image)['image']
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2YCrCb)[:,:,0]
        nrows,ncols = gt.shape
        if nrows != self.size_h or ncols != self.size_w:
            raise ValueError("Training image dimension is not the desired one.")
        gt = gt.astype(np.float32)
        gt /= 255.
        return gt

    def __len__(self,):
        return len(self.img_names)

class TestData(Dataset):
    def __init__(self,args):
        self.size = args.size
        self.data_path = args.benchmark_path
        self.data_list = os.listdir(self.data_path)

    def __getitem__(self,index):
        pic = cv2.imread(os.path.join(self.data_path,self.data_list[index]))
        image = cv2.cvtColor(pic,cv2.COLOR_BGR2YCrCb)[:,:,0]
        image_h, image_w = image.shape[:2]
        if image_h > image_w:
            image = cv2.flip(image, 1)
            image = cv2.transpose(image)
        gt = image
        return gt
    def __len__(self,):
        return len(self.data_list)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))


def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def Logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s')

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S')
    logfile = os.path.join(log_dir,localtime+'.log')
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 


def checkpoint(epoch, model, optimizer, model_out_path):
    torch.save({'pretrain_epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_out_path)

def load_checkpoint(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def load_checkpoint_double(model, pretrained_dict1, pretrained_dict2, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict1 = pretrained_dict1['state_dict']
    pretrained_model_dict2 = pretrained_dict2['state_dict']
    load_dict1 = {k: p for k, p in pretrained_model_dict1.items() if k in model_dict and p.shape == model_dict[k].shape}

    load_dict2 = {q: g for q, g in pretrained_model_dict2.items() if
                  q not in pretrained_model_dict1 and q in model_dict and g.shape == model_dict[q].shape}
    for name, param in model.named_parameters():
        if name in load_dict1:  # 如果是预训练参数
            param.requires_grad = False  # 固定参数，不参与训练
            print(f"{name}: requires_grad={param.requires_grad}")
    model_dict.update(load_dict1)
    model_dict.update(load_dict2)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict1), len(load_dict1)))
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict2), len(load_dict2)))


def load_checkpoint_double2(model, pretrained_dict1, pretrained_dict2, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict1 = pretrained_dict1['state_dict']
    pretrained_model_dict2 = pretrained_dict2['state_dict']

    load_dict1 = {k: p for k, p in pretrained_model_dict1.items() if k in model_dict and p.shape == model_dict[k].shape}

    # load_dict2 = {q: g for q, g in pretrained_model_dict2.items() if
    #               q in pretrained_model_dict1 and q in model_dict and g.shape == model_dict[q].shape}
    load_dict2 = {q: g for q, g in pretrained_model_dict2.items() if
                  q in model_dict and g.shape == model_dict[q].shape}
    # for name, param in model.named_parameters():
    #     if name in load_dict2:  # 如果是预训练参数
    #         param.requires_grad = False  # 固定参数，不参与训练
    #         print(f"{name}: requires_grad={param.requires_grad}")
    model_dict.update(load_dict1)
    model_dict.update(load_dict2)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict1), len(load_dict1)))
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict2), len(load_dict2)))


def load_checkpoint_double3(model, pretrained_dict1, pretrained_dict2, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict1 = pretrained_dict1['state_dict']
    pretrained_model_dict2 = pretrained_dict2['state_dict']
    load_dict1 = {k: p for k, p in pretrained_model_dict1.items() if k in model_dict and p.shape == model_dict[k].shape}
    load_dict2 = {k: p for k, p in pretrained_model_dict2.items() if k in model_dict and p.shape == model_dict[k].shape}
    # load_dict2 = {}
    # for k, v in pretrained_model_dict2.items():
        # if k.startswith('UE_block.'):
        #     # 替换键名前缀为denoiser2
        #     new_key = k.replace('UE_block.', '', 1)
        #     if new_key in model_dict and v.shape == model_dict[new_key].shape:
        #         load_dict2[new_key] = v

    # if 'mu' in pretrained_model_dict2 and pretrained_model_dict2['mu'].shape == model_dict['mu1'].shape:
    #     load_dict2['mu1'] = pretrained_model_dict2['mu']

    model_dict.update(load_dict1)
    model_dict.update(load_dict2)
    model.load_state_dict(model_dict)

    for name, param in model.named_parameters():
        if name in load_dict1:  # 如果是预训练参数
            param.requires_grad = False  # 固定参数，不参与训练
            # print(f"{name}: requires_grad={param.requires_grad}")

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict1), len(load_dict1)))
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict2), len(load_dict2)))


def load_checkpoint_withoutHW(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys() and k != 'H' and k != 'W'} # filtering parameters  and k != 'H' and k != 'W'
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def compute_ortho(A,B):
    h1, w1 = A.shape
    h2, w2 = B.shape
    AT = np.transpose(A)
    BT = np.transpose(B)
    return np.sqrt(MSE(np.matmul(A,AT), np.identity(h1))) + np.sqrt(MSE(np.matmul(B,BT), np.identity(h2)))


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimpleCA(nn.Module):
    def __init__(self, channel):
        super(SimpleCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

    def forward(self, x):
        y = self.sca(x)
        return x * y
