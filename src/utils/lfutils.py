from __future__ import absolute_import

import os
import skimage.io as io
import numpy as np
import math
import time
from skimage import color
from skimage.transform import resize


def FolderTo4DLF(path,ext,length):
    path_str = path+'/*.'+ext
    print('-'*40)
    print('Loading %s files from %s' % (ext, path) )
    img_data = io.ImageCollection(path_str)
    if len(img_data)==0:
        raise IOError('No .%s file in this folder' % ext)
    # print(len(img_data))
    # print img_data[3].shape
    N = int(math.sqrt(len(img_data)))
    if not(N**2==len(img_data)):
        raise ValueError('This folder does not have n^2 images!')

    [height,width,channel] = img_data[0].shape
    lf_shape = (N,N,height,width,channel)
    print('Initial LF shape: '+str(lf_shape))
    border = (N-length)/2
    if border<0:
        raise ValueError('Border {0} < 0'.format(border))
    out_lf_shape = (height, width, channel, length, length)
    print('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(config.floatX)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border,N-border,1):
        for j in range(border,N-border,1):
            indx = j + i*N
            im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            lf[:,:,0, i-border,j-border] = im[:,:,0]/255.0
            lf[:,:,1:3,i-border,j-border] = im[:,:,1:3]
            # io.imsave(save_path+str(indx)+'.png',img_data[indx])
    print('LF Range:')
    print('Channel 1 [%.2f %.2f]' %(lf[:,:,0,:,:].max(),lf[:,:,0,:,:].min()))
    print('Channel 2 [%.2f %.2f]' %(lf[:,:,1,:,:].max(),lf[:,:,1,:,:].min()))
    print('Channel 3 [%.2f %.2f]' %(lf[:,:,2,:,:].max(),lf[:,:,2,:,:].min()))
    print('--------------------')
    return lf


def AdjustTone(img,coef,norm_flag=False):

    print('--------------')
    print('Adjust Tone')

    tic = time.time()
    rgb = np.zeros(img.shape)
    img = np.clip(img,0.0,1.0)
    output = img ** (1/1.5)
    output = color.rgb2hsv(output)
    output[:,:,1] = output[:,:,1] * coef
    output = color.hsv2rgb(output)
    if norm_flag:
        r = output[:,:,0]
        g = output[:,:,1]
        b = output[:,:,2]
        rgb[:,:,0] = (r-r.min())/(r.max()-r.min())
        rgb[:,:,1] = (g-g.min())/(g.max()-g.min())
        rgb[:,:,2] = (b-b.min())/(b.max()-b.min())
    else:
        rgb = output

    print('IN Range: %.2f-%.2f' % (img.min(),img.max()))
    print('OUT Range: %.2f-%.2f' % (output.min(),output.max()))
    print("Elapsed time: %.2f sec" % (time.time() - tic))
    print('--------------')

    return  rgb

def modcrop(imgs,scale):

    if len(imgs.shape)==2:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row,:cropped_col]
    elif len(imgs.shape)==3:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row,:cropped_col,:]
    else:
        raise IOError('Img Channel > 3.')

    return  cropped_img

def ImgTo4DLF(filename,unum,vnum,length,adjust_tone,factor,save_sub_flag=False):

    if save_sub_flag:
        subaperture_path = os.path.splitext(filename)[0]+'_GT/'
        if not(os.path.exists(subaperture_path)):
            os.mkdir(subaperture_path)

    rgb_uint8 = io.imread(filename)
    rgb = np.asarray(skimage.img_as_float(rgb_uint8))
    print('Image Shape: %s' % str(rgb.shape))

    height = rgb.shape[0]/vnum
    width = rgb.shape[1]/unum
    channel = rgb.shape[2]

    if channel > 3:
        print('  Bands/Channels >3 Convert to RGB')
        rgb = rgb[:,:,0:3]
        channel = 3

    if adjust_tone > 0.0:
        rgb = AdjustTone(rgb,adjust_tone)

    cropped_height = height - height % factor
    cropped_width = width - width % factor

    lf_shape = (cropped_height, cropped_width, channel, vnum, unum)
    lf = np.zeros(lf_shape).astype(config.floatX)
    print('Initial LF shape: '+str(lf_shape))


    for i in range(vnum):
        for j in range(unum):
            im = rgb[i::vnum,j::unum,:]
            if save_sub_flag:
                subaperture_name = subaperture_path+'View_%d_%d.png' %(i+1,j+1)
                io.imsave(subaperture_name,im)
            lf[:,:,:,i,j] = color.rgb2ycbcr(modcrop(im,factor))
            lf[:,:,0,i,j] = lf[:,:,0,i,j]/255.0

    if unum % 2 == 0:
        border = (unum-length)/2 + 1
        u_start_indx = border
        u_stop_indx = unum - border + 1
        v_start_indx = border
        v_stop_indx = vnum - border + 1
    else:
        border = (unum-length)/2
        u_start_indx = border
        u_stop_indx = unum - border
        v_start_indx = border
        v_stop_indx = vnum - border

    if border<0:
        raise ValueError('Border {0} < 0'.format(border))

    out_lf = lf[:,:,:,v_start_indx:v_stop_indx,u_start_indx:u_stop_indx]
    print('Output LF shape: '+str(out_lf.shape))

    print('LF Range:')
    print('Channel 1 [%.2f %.2f]' %(out_lf[:,:,0,:,:].max(),out_lf[:,:,0,:,:].min()))
    print('Channel 2 [%.2f %.2f]' %(out_lf[:,:,1,:,:].max(),out_lf[:,:,1,:,:].min()))
    print('Channel 3 [%.2f %.2f]' %(out_lf[:,:,2,:,:].max(),out_lf[:,:,2,:,:].min()))
    print('--------------------')

    bic_lf = np.zeros(out_lf[:,:,0,:,:].shape).astype(config.floatX)

    for i in range(bic_lf.shape[2]):
        for j in range(bic_lf.shape[3]):
            this_im = out_lf[:,:,0,i,j]
            lr_im = resize(this_im, (cropped_height/factor,cropped_width/factor),
                           order=3, mode='symmetric', preserve_range=True)
            bic_lf[:,:,i,j] = resize(lr_im, (cropped_height,cropped_width),
                                     order=3, mode='symmetric', preserve_range=True)

    return out_lf, bic_lf