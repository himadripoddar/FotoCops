import os
import numpy as np
import cv2
import requests
import sys

from PIL import Image
from io import BytesIO
from matplotlib import pyplot


import modelCore
manTraNet = modelCore.load_pretrain_model_by_index( 4, './pretrained_weights')

from datetime import datetime
def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb

def decode_an_image_array( rgb, manTraNet, dn=1 ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )[:,::dn,::dn]
    t0 = datetime.now()
    y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet, dn=1 ) :
    rgb = read_rgb_image( image_file )
    mask, ptime = decode_an_image_array( rgb, manTraNet, dn )
    return rgb[::dn,::dn], mask, ptime.total_seconds()

def get_image_from_url(url, xrange=None, yrange=None, dn=1) :
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    if img.shape[-1] > 3 :
        img = img[...,:3]
    ori = np.array(img)
    if xrange is not None :
        img = img[:,xrange[0]:xrange[1]]
    if yrange is not None :
        img = img[yrange[0]:yrange[1]]
    mask, ptime =  decode_an_image_array( img, manTraNet, dn )
    ptime = ptime.total_seconds()
    # show results
    pyplot.figure( figsize=(15,5) )
    pyplot.title('Original Image')
    pyplot.subplot(131)
    pyplot.imshow( img )
    pyplot.title('Forged Image (FotoCops)')
    pyplot.subplot(132)
    pyplot.imshow( mask, cmap='gray' )
    pyplot.title('Predicted Mask (FotoCops)')
    pyplot.subplot(133)
    pyplot.imshow( np.round(np.expand_dims(mask,axis=-1) * img[::dn,::dn]).astype('uint8'), cmap='jet' )
    pyplot.title('Highlighted Forged Regions')
    # pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, rgb.shape, ptime ) )
    pyplot.show()

def get_image_from_local(path, xrange=None, yrange=None, dn=1) :
    img = Image.open(path)
    img = np.array(img)
    if img.shape[-1] > 3 :
        img = img[...,:3]
    ori = np.array(img)
    if xrange is not None :
        img = img[:,xrange[0]:xrange[1]]
    if yrange is not None :
        img = img[yrange[0]:yrange[1]]
    mask, ptime =  decode_an_image_array( img, manTraNet, dn )
    ptime = ptime.total_seconds()
    # show results
    pyplot.figure( figsize=(15,5) )
    pyplot.title('Original Image')
    pyplot.subplot(131)
    pyplot.imshow( img )
    pyplot.title('Forged Image (FotoCops)')
    pyplot.subplot(132)
    pyplot.imshow( mask, cmap='gray' )
    pyplot.title('Predicted Mask (FotoCops)')
    pyplot.subplot(133)
    pyplot.imshow( np.round(np.expand_dims(mask,axis=-1) * img[::dn,::dn]).astype('uint8'), cmap='jet' )
    pyplot.title('Highlighted Forged Regions')
    # pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, rgb.shape, ptime ) )
    pyplot.show()
# get_image_from_url('https://www.stockvault.net/blog/wp-content/uploads/2015/08/july-2.jpg')
get_image_from_local('./data/forged/I00_dgzb67i_0.jpg')
