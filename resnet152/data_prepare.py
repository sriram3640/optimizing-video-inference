from PIL import Image
import glob
import os
import numpy as np
import csv
import random
from skimage.transform import resize
from skimage import io
from sklearn.model_selection import train_test_split
import re

image_size = 224

r1 = re.compile("frame_(\d+)")
r2 = re.compile("patch_(\d+)")
def key1(a):
    m1 = r1.findall(a)
    m2 = r2.findall(a)
    return int(m1[0]), int(m2[0])

#resize images
def resize_Images():
    for filepath in glob.iglob('data/original/*.jpg'):
        filename = os.path.basename(filepath)

        image = io.imread(filepath)
        resized = resize(image, (image_size, image_size), anti_aliasing=True)        
        io.imsave('dataset/resized224/' + filename, resized)

        #with open(filepath, 'r+b') as f:
            #with Image.open(f) as image:
                #resized = resizeimage.resize_crop(image, [image_size, image_size])
                #resized.save('dataset/resized224/' + filename, image.format)
        

def load_Data():
    
    (x_all, y_all) = load_batch()
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=10)    
    return (x_train, y_train), (x_test, y_test)

def load_batch():
    #image data
    imageCount = 0
    filelist = []
    for imagefile in glob.iglob('data/resized224/*.jpg'):
        filelist.append(imagefile)
        imageCount += 1

    filelist.sort(key = key1)
    data = np.zeros((imageCount, image_size, image_size, 3))
    labels = np.zeros(imageCount)
    index = 0

    for counter, imagefile in enumerate(filelist):
        #print(str(counter + 1) + 'load image file: ' + imagefile)
        t = Image.open(imagefile)
        arr = np.array(t) #Convert test image into an array 32*32*3    
        data[index] = arr 
        index += 1
        
    #labels
    labels = []
    #for filepath in glob.iglob(''):
    with open('data/labels/all.txt') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            labels.append(line.strip().split(',')[1])    
    return data, labels
