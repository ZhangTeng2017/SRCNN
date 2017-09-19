# -*- coding: UTF-8 -*-
import os
import numpy as np
import math
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import h5py
# Make sure that caffe is on the python path:
caffe_root = '/home/sdb/zhangteng/git/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.append(caffe_root+'python')
sys.path.append(caffe_root+'python/caffe')
import caffe


# Parameters
scale = 3

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('dat'))
    label = np.array(hf.get('lab'))
    return data, label

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

def SRCNN(net,im_b):
	#get params
	weights_conv1=net.params['conv1'][0].data
	biases_conv1=net.params['conv1'][1].data
	weights_conv2=net.params['conv2'][0].data
	biases_conv2=net.params['conv2'][1].data
	weights_conv3=net.params['conv3'][0].data
	biases_conv3=net.params['conv3'][1].data
	#process weights
	conv1_filters, _ ,conv1_patchsize, _ = weights_conv1.shape
	conv2_filters,conv2_channels,conv2_patchsize,_ =weights_conv2.shape
	_ , conv3_channels,conv3_patchsize,_ =weights_conv3.shape
	width,height = im_b.shape
	#conv1
	weights_conv1 = np.squeeze(weights_conv1)
	conv1_data = np.zeros((width,height,conv1_filters),dtype=np.float64)
	for i in range(0,conv1_filters,1):
		weights_conv1[i,:,:]=weights_conv1[i,:,:].T #和matlab是行列相反
		conv1_data[:,:,i]=cv2.filter2D(im_b,-1,weights_conv1[i,:,:])
		conv1_data[:,:,i]=conv1_data[:,:,i]+biases_conv1[i]
		index=conv1_data<0
		conv1_data[index]=0
	#conv2
	weights_conv2 = np.squeeze(weights_conv2)
	conv2_data = np.zeros((width,height,conv2_filters),dtype=np.float64)
	for i in range(0,conv2_filters,1):
		for j in range(0,conv2_channels,1):
			conv2_subfilter = weights_conv2[i,j].reshape([conv2_patchsize,conv2_patchsize])
			conv2_data[:,:,i] = conv2_data[:,:,i] + cv2.filter2D(conv1_data[:,:,j],-1,conv2_subfilter)
		conv2_data[:,:,i] = conv2_data[:,:,i]+biases_conv2[i]
		index=conv2_data<0
		conv2_data[index]=0
	#conv3
	weights_conv3 = np.squeeze(weights_conv3)
	conv3_data = np.zeros((width,height),dtype=np.float64)
	for i in range(0,conv3_channels,1):
		weights_conv3[i,:,:]=weights_conv3[i,:,:].T #和matlab是行列相反
		conv3_subfilter = weights_conv3[i,:,:]
		conv3_data = conv3_data + cv2.filter2D(conv2_data[:,:,i],-1,conv3_subfilter)
	conv3_data[:,:] = conv3_data[:,:]+biases_conv3
	return conv3_data


def shave(image,scale):
	image = image[scale:-scale,scale:-scale]
	return image


def Predict(deploy_proto,caffe_model,test_image,h5_image,results_path):

	net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
	#read image
	image = cv2.imread(test_image)
	image = modcrop(image,scale)
	ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

	lr_img,hr_img = read_data(h5_image)
	lr_img=lr_img.T #和matlab是行列相反
	hr_img=hr_img.T #和matlab是行列相反
	shape = hr_img.shape

	pred_img=SRCNN(net,lr_img)
	pred_img = shave((np.rint(pred_img[:,:] * 255)).astype(np.uint8),scale)
	hr_img = shave((np.rint(hr_img[:,:]*255)).astype(np.uint8),scale)
	lr_img = shave((np.rint(lr_img[:,:]*255)).astype(np.uint8),scale)

	#Show color image
	ycrcb = ycrcb[scale:-scale,scale:-scale,:]
	pred_img = colorize(pred_img, ycrcb)
	lr_img = colorize(lr_img, ycrcb)
	hr_img =colorize(hr_img,ycrcb)

	#PSNR
	# print("SRCNN结果:")
	psnr_final=PSNR(pred_img,hr_img)
	# print("bicubic结果:")
	psnr_basic=PSNR(lr_img,hr_img)

	#save

	name = test_image.split("/")[-1]
	name = os.path.splitext(name)[0]
	cv2.imwrite(results_path+name+'_pred.png', pred_img)
	cv2.imwrite(results_path+name+'_hr.png', hr_img)
	cv2.imwrite(results_path+name+'_lr.png', lr_img)

	return psnr_basic , psnr_final

if __name__ == "__main__":
	deploy_proto=caffe_root + 'examples/SR1/SRCNN_deploy.prototxt'
	caffe_model=caffe_root + 'examples/SR1/SRCNN_iter_300000.caffemodel'
	test_path = caffe_root + 'examples/SR1/Test/Set5/'
	h5_path = caffe_root + 'examples/SR1/Test_h5/Set5/'
	results_path = caffe_root + 'examples/SR1/Result/'

	image_names = os.listdir(test_path)
	image_names = sorted(image_names)
	h5_names = os.listdir(h5_path)
    	h5_names = sorted(h5_names)
    	nums = image_names.__len__()
    	bicubic = []
    	srcnn = []
    	names =[]
    	for i in range(nums):
    		test_image = test_path + image_names[i]
    		h5_image = h5_path + h5_names[i]
		name = os.path.splitext(image_names[i])[0]
		bi,sr=Predict(deploy_proto,caffe_model,test_image,h5_image,results_path)
		bicubic.append(bi)
		srcnn.append(sr)
		names.append(name)
	print names 
	print bicubic 
	print srcnn


		
