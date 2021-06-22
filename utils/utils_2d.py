import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import skimage.transform
import utils.diffuser_utils as df 
import cv2
import torch
import utils.common_utils as cu

def load_psf(path, f): # Function to load PSF 
    psf = np.array(Image.open(path))
    psf_bg = np.mean(psf[0 : 15, 0 : 15])             #102
    psf = skimage.transform.resize(psf-psf_bg, (psf.shape[0]//f,psf.shape[1]//f), anti_aliasing = True)
    psf = psf/np.linalg.norm(psf)
    return np.flipud(psf)

def guass_fn(img):  # Apply gaussian fall-off to image 
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols,200)
    kernel_y = cv2.getGaussianKernel(rows,200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    return img*np.stack((mask,mask,mask), axis=-1)

def load_img(path):
    img = np.flipud(df.flip_channels(np.load(path)))
    return img/np.max(img)
def get_eraser(diffuser, erase_rate=0.5):
    w, h = diffuser.shape[0], diffuser.shape[1]  
    zeros = int(erase_rate*w*h)
    ones = w*h - zeros   
    arr = np.array([0] * zeros + [1] * ones)
    np.random.shuffle(arr)
    arr = arr.reshape([w,h])    
    arr_3d = np.stack((arr,arr,arr), axis=2)
    return arr_3d
def apply_eraser(mask, ts, arr):
    eraser_ts = torch.tensor(mask.transpose(2,0,1), dtype=torch.float32).cuda().unsqueeze(0)
    diffuser_torch_er = ts * eraser_ts
    diffuser_torch_er /= torch.max(diffuser_torch_er)
    load_diffuser_np_er = arr*mask
    load_diffuser_np_er /= np.max(load_diffuser_np_er)
    return diffuser_torch_er, load_diffuser_np_er

def preplot(recons):
    recons = cu.ts_to_np(recons).transpose(1,2,0)
    recons /= np.max(recons)
    return recons[recons.shape[0]//4:-recons.shape[0]//4,recons.shape[1]//4:-recons.shape[1]//4]
def plot(groundtruth, recons):
    plt.figure()
    plt.subplot(121)
    plt.title('Groundtruth')
    plt.axis('off')
    plt.imshow(groundtruth)
    plt.subplot(122)
    plt.title('Reconstruction')
    plt.axis('off')
    plt.imshow(preplot(recons))
    plt.show()