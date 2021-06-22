import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import utils.common_utils as cu

def load_data(img_path = 'data/single_shot_hyperspectral/meas_thordog.png', ds = 2, ds_spec = True, simulated = True):
    
    
    loaded_mat = scipy.io.loadmat('data/single_shot_hyperspectral/calibration.mat')

    # Pre-process mask and PSF
    mask1 = np.asarray(loaded_mat['mask'], dtype = np.float32)
    psf1 = np.asarray(loaded_mat['psf'], dtype = np.float32)

    c1 = 100; c2 = 420; c3 = 80; c4 = 540
    mask = mask1[c1:c2, c3:c4, :]
    psf = psf1[c1:c2, c3:c4]

    psf = psf/np.linalg.norm(psf)

    mask_sum = np.sum(mask, 2)
    ind = np.unravel_index((np.argmax(mask_sum, axis = None)), mask_sum.shape)
    mask[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2, :] = 0
    
    # Load in image
    img = plt.imread(img_path)
    
    im = np.asarray(img[c1:c2, c3:c4]) #.astype(np.float32)
    im = im/np.max(im)
    im[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2] = 0
    
    mask = mask[:,:,1:]
    if ds_spec:
        mask = (mask[:,:,0::2]+mask[:,:,1::2])/2
    psf = psf[:-20,20:-40]
    mask = mask[:-20,20:-40]
    im = im[:-20,20:-40]
    
    
    im  = cv2.resize(im, (psf.shape[1]//ds, psf.shape[0]//ds))
    mask = cv2.resize(mask, (psf.shape[1]//ds, psf.shape[0]//ds))
    psf = cv2.resize(psf, (psf.shape[1]//ds, psf.shape[0]//ds))

    if simulated == True:
        gt = scipy.io.loadmat('data/single_shot_hyperspectral/sim_fruit_gt.mat')['gt_im']
    else:
        gt = None

    return im, mask, psf, gt

def stack_rgb_opt(reflArray, opt = 'utils/false_color_calib.mat', scaling = [1,1,2.5]):
    
    color_dict = scipy.io.loadmat(opt)
    red = color_dict['red']; green = color_dict['green']; blue = color_dict['blue']
    
    reflArray = reflArray/np.max(reflArray)
    
    red_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    green_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    blue_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    
    num_channels = reflArray.shape[-1]
    
    if num_channels == 32:
        for i in range(0,32):
            red_channel = red_channel + reflArray[:,:,i]*red[0,i*2]*scaling[0]
            green_channel = green_channel + reflArray[:,:,i]*green[0,i*2]*scaling[1]
            blue_channel = blue_channel + reflArray[:,:,i]*blue[0,i*2]*scaling[2]
    else:
        for i in range(0,64):
            red_channel = red_channel + reflArray[:,:,i]*red[0,i]*scaling[0]
            green_channel = green_channel + reflArray[:,:,i]*green[0,i]*scaling[1]
            blue_channel = blue_channel + reflArray[:,:,i]*blue[0,i]*scaling[2]

    red_channel = red_channel/num_channels
    green_channel = green_channel/num_channels
    blue_channel = blue_channel/num_channels

    stackedRGB = np.stack((red_channel,green_channel,blue_channel),axis=2)

    return stackedRGB

def preplot(recons):
    recons = cu.ts_to_np(recons).transpose(1,2,0)
    recons = np.flipud(np.fliplr(recons))
    recons = stack_rgb_opt(recons)
    recons /= np.max(recons)
    return recons

def preplot2(recons):
    recons = cu.ts_to_np(recons).transpose(1,2,0)
    recons = np.flipud(np.fliplr(recons))
    return recons

def plot(recons):
    recons = preplot(recons)
    #plt.imshow(np.mean(recons,-1), cmap='gray')
    plt.imshow(recons)
    plt.title('Reconstruction: false color projection')
    plt.show()