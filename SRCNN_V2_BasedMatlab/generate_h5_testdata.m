clear all;
close all;
%% settings
folder = 'Test/Set5';
savepath= 'Test_h5/Set5/';
savename = {'baby_GT.h5','bird_GT.h5','butterfly_GT.h5','head_GT.h5','woman_GT.h5'};
scale = 3;
%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1:5 
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
    
    data(:, :) = im_input;
    label(:, : ) = im_label;
    
    path=strcat(savepath,char(savename(i)));
    %% writing to HDF5
    h5create(path,'/dat',size(data));
    h5create(path,'/lab',size(label));
    h5write(path,'/dat',data)
    h5write(path,'/lab',label)
    clear data
    clear label
end


