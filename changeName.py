# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:08:39 2019

@author: Raytine

change pic name
"""

import os
import cv2
from tqdm import tqdm

input_size = (512, 512)

def change_name():
    path = '.\\data\\nuclei\\train\\label'
    fileList=os.listdir(path)
    for i in range(len(fileList)):
        #设置旧文件名（就是路径+文件名）
        oldname = path+ os.sep + fileList[i]   # os.sep添加系统分隔符
        #设置新文件名
        newname = '_'.join(oldname.split('_')[:-1]) + '.png'
        os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
        print(oldname,'======>',newname)
        
def img_crops(filepath, destpath):
    pathDir =  os.listdir(filepath) # list all the path or file  in filepath
    for allDir in tqdm(pathDir):
        child = os.path.join(filepath, allDir)
        dest = os.path.join(destpath,allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
            sp = image.shape#obtain the image shape
            sz1 = sp[0]#height(rows) of image
            sz2 = sp[1]#width(colums) of image
            #你想对文件的操作
            for i in range(0,sz1-input_size[0]+1, 244):
                for j in range(0,sz2-input_size[1]+1, 244):
                    cropImg = image[i:i+input_size[0],j:j+input_size[1]] #crop the image
                    dest_temp = '.'.join(dest.split('.')[:-1])+'_%s_%s.png' % (i,j)
                    cv2.imwrite(dest_temp, cropImg)# write in destination path

if __name__ == '__main__':
    filepath = '.\\data\\MoNuSeg\\train\\label'
    destpath = '.\\data\\MoNuSeg\\train\\label_crop'
    img_crops(filepath, destpath)