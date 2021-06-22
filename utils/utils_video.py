import numpy as np
from PIL import Image
import skimage
import skimage.transform
import scipy.io as io
import matplotlib.pyplot as plt
import utils.common_utils as cu
import scipy.io

def load_data(path, f):
    img = np.array(Image.open(path))
    img = skimage.transform.resize(img, (img.shape[0]//f,img.shape[1]//f), anti_aliasing = True)
    img /= np.max(img)
    return img
def load_mask(path, target_shape):
    shutter = io.loadmat(path)['shutter_indicator']
    shutter = shutter[shutter.shape[0]//4:-shutter.shape[0]//4,shutter.shape[1]//4:-shutter.shape[1]//4,:]
    shutter = skimage.transform.resize(shutter, (target_shape[0],target_shape[1]), anti_aliasing = True)
    return shutter

def load_simulated():
    downsampling_factor = 16
    mask_np = scipy.io.loadmat('data/single_shot_video/shutter_ds.mat')['shutter_indicator'][1:-2,...]
    meas_np = scipy.io.loadmat('data/single_shot_video/meas_simulated.mat')['im']
    mask_np = mask_np[meas_np.shape[0]//2:-meas_np.shape[0]//2, meas_np.shape[1]//2:-meas_np.shape[1]//2]
    psf_np = load_data('data/single_shot_video/psf.tif',downsampling_factor)[1:][...,1]
    return meas_np, psf_np, mask_np

def preplot(recons):
    recons = cu.ts_to_np(recons).transpose(1,2,0)
    recons /= np.max(recons)
    return recons[recons.shape[0]//4:-recons.shape[0]//4,recons.shape[1]//4:-recons.shape[1]//4]

def preplot2(recons):
    recons = cu.ts_to_np(recons).transpose(2,3,0,1)
    recons /= np.max(recons)
    recons = np.clip(recons, 0,1)
    return recons

def plot(channel, recons):
    recons = preplot(recons)
    #n = random.randint(0,recons.shape[-1]-1)
    #frame = recons[:,:,n]
    plt.imshow(np.mean(recons,-1), cmap='gray')
    plt.title('Reconstruction: channel %d mean projection'%(channel))
    plt.show()
def plot3d(recons):
    recons = recons[0].detach().cpu().numpy().transpose(2,3,0,1)
    #n = random.randint(0,recons.shape[-1]-1)
    #frame = recons[:,:,n]
    plt.imshow(np.mean(recons,-1))
    plt.title('Reconstruction: mean projection')
    plt.show()
def plot_slider(x):
    plt.title('Reconstruction: frame %d'%(x))
    plt.axis('off')
    plt.imshow(video[:,:,:,x])
    return x