import torch.nn.functional as F
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from PIL import Image 
import skimage
import skimage.transform
import glob
import matplotlib.pyplot as plt

def flip_channels(image):
    image_color = np.zeros_like(image); 
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    return(image_color)

def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, :, C01:C02, C11:C12]

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

###### Complex operations ##########
def complex_multiplication(t1, t2):

    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def complex_abs(t1):
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)

def make_real(c):
    out_r, _ = torch.unbind(c,-1)
    return out_r

def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)

def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    x = x.cuda()
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(dh[:, :, :-1] + dw[:, :, :, :-1] )

class Forward_Model(nn.Module):
    def __init__(self, h, eraser=None, cuda_device = 0):
        super(Forward_Model, self).__init__()

        self.cuda_device = cuda_device
         ## Initialize constants 
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions
        
        self.PAD_SIZE0 = int((self.DIMS0)//2)                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)                           # Pad size
        
        self.h_var = torch.nn.Parameter(torch.tensor(h, dtype=torch.float32, device=self.cuda_device),
                                            requires_grad=False)
            
        self.h_zeros = torch.nn.Parameter(torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float32, device=self.cuda_device),
                                          requires_grad=False)

        self.h_complex = torch.stack((pad_zeros_torch(self, self.h_var), self.h_zeros),2).unsqueeze(0)
        
        self.const = torch.tensor(1/np.sqrt(self.DIMS0*2 * self.DIMS1*2), dtype=torch.float32, device=self.cuda_device)
        
        self.H = torch.fft(batch_ifftshift2d(self.h_complex).squeeze(), 2)   
        
        self.eraser = torch.tensor(eraser.transpose(2,0,1), dtype=torch.float32, device=self.cuda_device).unsqueeze(0)
        
    def Hfor(self, x):
        xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
        X = torch.fft(xc,2)
        HX = complex_multiplication(self.H,X)
        out = torch.ifft(HX,2)
        out_r, _ = torch.unbind(out,-1)
        return out_r
        
    def forward_zero_pad(self, in_image):
        output = self.Hfor(pad_zeros_torch(self,in_image))
        return crop(self,output)*self.eraser
    
    def forward(self, in_image):
        output = self.Hfor(in_image)
        return crop(self, output)*self.eraser
    
    
def my_pad(model, x):
    PADDING = (model.PAD_SIZE1//2, model.PAD_SIZE1//2, model.PAD_SIZE0//2, model.PAD_SIZE0//2)
    
    return F.pad(x, PADDING, 'constant', 0)
def crop_forward(model, x):
    C01 = model.PAD_SIZE0//2; C02 = model.PAD_SIZE0//2 + model.DIMS0//2              # Crop indices 
    C11 = model.PAD_SIZE1//2; C12 = model.PAD_SIZE1//2 + model.DIMS1//2              # Crop indices 
    return x[..., C01:C02, C11:C12]

#def crop_forward2(model, x):
#    C01 = model.PAD_SIZE0//2; C02 = model.PAD_SIZE0//2 + model.DIMS0//2              # Crop indices 
#    C11 = model.PAD_SIZE1//2; C12 = model.PAD_SIZE1//2 + model.DIMS1//2              # Crop indices 
#    return x[:, :, :, C01:C02, C11:C12]
class Forward_Model_combined(torch.nn.Module):
    def __init__(self, h_in, imaging_type = '2D', shutter=0, cuda_device = 0):
        super(Forward_Model_combined, self).__init__()

        self.cuda_device = cuda_device
        self.imaging_type = imaging_type
         ## Initialize constants 
        self.DIMS0 = h_in.shape[0]  # Image Dimensions
        self.DIMS1 = h_in.shape[1]  # Image Dimensions
        
        self.PAD_SIZE0 = int((self.DIMS0)//2)                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)                           # Pad size
        
        h = h_in
            
        self.H = torch.stack((torch.tensor(np.real(h),dtype=torch.float32, device=self.cuda_device), 
                      torch.tensor(np.imag(h),dtype=torch.float32, device=self.cuda_device)),-1).unsqueeze(0)
        
        self.shutter = np.transpose(shutter, (2,0,1))
        self.shutter_var = torch.tensor(self.shutter, dtype=torch.float32, device=self.cuda_device).unsqueeze(0)

        
    def Hfor(self, x):
        xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
        X = torch.fft(xc,2)
        HX = complex_multiplication(self.H,X)
        out = torch.ifft(HX,2)
        out_r, _ = torch.unbind(out,-1)
        return out_r
    
        
    def forward(self, in_image):
        
        if self.imaging_type == 'spectral' or self.imaging_type == 'video':
            if len(in_image.shape)==5:
                output = torch.sum(self.shutter_var.unsqueeze(0) * crop_forward(self,  self.Hfor(my_pad(self, in_image))), 2)
            else:
                output = torch.sum(self.shutter_var * crop_forward(self,  self.Hfor(my_pad(self, in_image))), 1)
        elif self.imaging_type == '2D_erasures':
            output = self.shutter_var*crop_forward(self,  self.Hfor(in_image))
        elif self.imaging_type == '2D':
            output = crop_forward(self,  self.Hfor(in_image))
        else:
            output = torch.sum(crop_forward(self,  self.Hfor(in_image)), 1)
        return output