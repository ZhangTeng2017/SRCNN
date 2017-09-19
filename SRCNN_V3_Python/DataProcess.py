# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import scipy.misc
import numpy as np


DATA_PATH = "/home/sdb/zhangteng/git/caffe/examples/SR/Train/"
TEST_PATH = "/home/sdb/zhangteng/git/caffe/examples/SR/Test/Set5/"
image_size = 33
label_size = 21
stride_train = 14
stride_test = 21
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

def prepare_data(_path,stride):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    sub_input_sequence = []
    sub_label_sequence = []

    padding = abs(image_size - label_size)/2

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCR_CB)
        hr_img = hr_img[:, :, 0]
        hr_img = modcrop(hr_img,scale)
        hr_img=hr_img/255.
        h,w= hr_img.shape
        # two resize operation to produce training data and labels
        lr_img = cv2.resize(cv2.resize(hr_img, (w/3, h/3), interpolation=cv2.INTER_AREA),
        	(w, h), interpolation=cv2.INTER_CUBIC)
        for x in range(0,h-image_size+1,stride):
            for y in range(0,w-image_size+1,stride):
                sub_input = lr_img[x:x+image_size, y:y+image_size]  # [33 x 33]
                sub_label = hr_img[x+padding:x+padding+label_size, y+padding:y+padding+label_size] # [21 x 21]

                # Make channel value
                sub_input = sub_input.reshape([1,image_size, image_size])  
                sub_label = sub_label.reshape([1,label_size, label_size])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    # Make list to numpy array. With this transform
    data = np.asarray(sub_input_sequence)  # [?,1,33,33]
    label = np.asarray(sub_label_sequence) # [?,1,21,21]

    return data, label

def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """
    x = data.astype(np.float64)
    y = labels.astype(np.float64)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


if __name__ == "__main__":
    data, label = prepare_data(DATA_PATH,stride_train)
    write_hdf5(data, label, "train.h5")
    print data.shape
    print label.shape
    data, label = prepare_data(TEST_PATH,stride_test)
    write_hdf5(data, label, "test.h5")
    print data.shape
    print label.shape
