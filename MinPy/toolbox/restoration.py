# TODO translate skimage.restoration into the tensor version
import bm3d
from skimage.restoration import estimate_sigma
import numpy as np

def bm3d_denoise(img):
    sigma = np.mean(estimate_sigma(img))
    bm3d_img = bm3d.bm3d(img,sigma)
    bm3d_img = np.clip(bm3d_img,0,1)
    return bm3d_img
