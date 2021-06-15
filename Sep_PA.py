#!/usr/bin/env python
# coding: utf-8

# In[3]:


from astropy.io import fits
import trackpy as tp
import numpy as np
from scipy import ndimage, misc


# In[4]:
pixscale = 0.01227

def circle_mask(im, xc, yc, rcirc):
    """circle_mask - This function creates a circle mask with radius rcirc with center (xc,yc) in a 2D array the same as the input image im"""
    xlen = len(im)
    ylen = len(im)
    yv,xv = np.mgrid[0:ylen,0:xlen]
    Mask = ((xv-xc)**2 + (yv-yc)**2 < rcirc**2)
    return Mask

def circle_mask2(im, xc, yc, rcirc):
    """circle_mask - This function creates a circle mask with radius rcirc with center (xc,yc) in a 2D array the same as the input image im"""
    xlen = len(im[0])
    ylen = len(im)
    yv,xv = np.mgrid[0:ylen,0:xlen]
    Mask = ((xv-xc)**2 + (yv-yc)**2 > rcirc**2)
    return Mask
# In[5]:

def calc_sep_pa3(data, dia1, dia2, radmask, sep, noise, ang):
    """
    Function to return the separation, position angle, and rotated image (due to true north correction SPHERE) of point sources in a
    data reduced image. 
    
    :param data : the not yet unpacked data array, after importing it with get_pkg_data_filename().
    :param dia1, dia2 : the diameters of the point sources you are looking for. They need to be uneven integers starting from 3 (smallest 
    possible diameter). There are two to prevent double detections, as trackpy checks for whether it is detected for both diameters. Is in 
    pixels (float).
    :param radmask : the radius of the mask you want to apply over the star, as the stellar PSF is not perfectly subtracted, it will results
    in a lot of detections from trackpy, which are not actual point sources. Is in pixels as well (float).
    :param sep : the minimum separation between detected sources, tweak this to ensure no double detections. It is in number of pixels. 
    :param noise : the expected size of the noise in counts. Tweak this to also get low count point sources as detections.
    :param ang : the angle with which you want to rotate the data array to account for some sort of correction in degrees (deg). For SPHERE
    this is 1.75 degrees. Put 0 if you don't want to rotate the array. 
    """
    dat = fits.getdata(data, ext=0)
    dat1 = dat[0]
    image_rot = ndimage.rotate(dat1, ang, reshape=True)
    dat2 = image_rot.astype(int)
    radnew = len(dat2[0][:])/2.
    MaskedImage =dat2 * (1-circle_mask(dat2, radnew, radnew, radmask))
    f = tp.locate(MaskedImage, (rad1,rad2), separation = float(sep), noise_size = noise)
    offset_x = f[:].x - 528.
    offset_y = f[:].y - 528.
    sep = np.sqrt(offset_x**2 + offset_y**2)*pixscale
    pa = np.arctan2(offset_x, offset_y)
    return sep, pa, image_rot