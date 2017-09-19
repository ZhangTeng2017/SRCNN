# -*- coding: UTF-8 -*-
import os
import numpy as np
import math
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/sdb/zhangteng/git/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.append(caffe_root+'python')
sys.path.append(caffe_root+'python/caffe')
import caffe

# Parameters
scale = 3


def modcrop(image,scale):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def colorize(y, ycrcb):
    y[y>255] = 255
    
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    
    return img

# PSNR measure, from ANR's code
def PSNR(pred, gt):
    f = pred.astype(float)
    g = gt.astype(float)
    e = (f - g).flatten()
    rmse = math.sqrt(np.mean(e ** 2.))
    return 20 * math.log10(255. / rmse)


def Predict(deploy_proto,caffe_model,test_image,results_path):

	caffe.set_device(1);
	net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
	# Inputs
	hr_img = cv2.imread(test_image)
	hr_img = modcrop(hr_img,scale)
	ycrcb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCR_CB)
	hr_img = ycrcb[:,:,0]
	hr_img = hr_img /255.
	h,w=hr_img.shape
	lr_img = cv2.resize(cv2.resize(hr_img, (w/3, h/3), interpolation=cv2.INTER_AREA),
		(w, h), interpolation=cv2.INTER_CUBIC)

	net.blobs['data'].reshape(1, 1, h, w)
	net.blobs['data'].data[...] = np.reshape(lr_img , (1, 1, h, w))
	net.forward()
	out = net.blobs['conv3'].data[...]

	pred_img= np.squeeze(out)
	pred_img = (np.rint(pred_img[:,:] * 255)).astype(np.uint8)
	hr_img = hr_img[6:-6,6:-6]
	hr_img = (np.rint(hr_img[:,:]*255)).astype(np.uint8)
	lr_img = lr_img[6:-6,6:-6]
	lr_img = (np.rint(lr_img[:,:]*255)).astype(np.uint8)
	
	#Show color image
	ycrcb=ycrcb[6:-6,6:-6,:]
	pred_img = colorize(pred_img, ycrcb)
	lr_img = colorize(lr_img, ycrcb)
	hr_img =colorize(hr_img,ycrcb)

	#PSNR
	# print("SRCNN结果:")
	psnr_final=PSNR(pred_img,hr_img)
	# print("bicubic结果:")
	psnr_basic=PSNR(lr_img,hr_img)

	name = test_image.split("/")[-1]
	name = os.path.splitext(name)[0]
	cv2.imwrite(results_path+name+'_pred.png', pred_img)
	cv2.imwrite(results_path+name+'_hr.png', hr_img)
	cv2.imwrite(results_path+name+'_lr.png', lr_img)

	return psnr_basic , psnr_final

	
if __name__ == "__main__":
	deploy_proto=caffe_root + 'examples/SR/SR_deploy.prototxt'
	caffe_model=caffe_root + 'examples/SR/model/SR_iter_1130000.caffemodel'
	test_path = caffe_root + 'examples/SR/Test/Set5/'
	results_path = caffe_root + 'examples/SR/result/'

	image_names = os.listdir(test_path)
	image_names = sorted(image_names)
	nums = image_names.__len__()
    	bicubic = []
    	srcnn = []
    	names =[]
    	for i in range(nums):
    		test_image = test_path + image_names[i]
    		name = os.path.splitext(image_names[i])[0]
    		bi,sr=Predict(deploy_proto,caffe_model,test_image,results_path)
    		bicubic.append(bi)
		srcnn.append(sr)
		names.append(name)
	print names 
	print bicubic 
	print srcnn