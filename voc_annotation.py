import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
# To increase the size of the test set, modify the trainval_percent parameter.
# Modify the train_percent parameter to change the validation set proportion to 9:1.
# Currently, this library uses the test set as the validation set, and it does not separate the test set.
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
# Point to the folder where the VOC dataset is located.
# By default, it points to the VOC dataset in the root directory.
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("Check if the dataset format meets the requirements; this may take some time.")
    classes_nums        = np.zeros([256], np.int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not detected. Please check if the file exists in the specified path and if the file extension is .png."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The shape of label image %s is %s, and it does not appear to be a grayscale image or an 8-bit color image. Please carefully review the dataset format."%(name, str(np.shape(png))))
            print("Label images should be either grayscale or 8-bit color images, where each pixel's value represents the category to which that pixel belongs."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print the pixel values and their corresponding counts.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It has been detected that the pixel values in the labels only include 0 and 255. The data format appears to be incorrect.")
        print("For binary classification tasks, you should modify the labels so that background pixels have a value of 0, and target pixels have a value of 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("It has been detected that the labels only contain background pixels. The data format appears to be incorrect. Please carefully review the dataset format.")

    print("The images in the JPEGImages folder should be in the .jpg file format, while the images in the SegmentationClass folder should be in the .png file format.")