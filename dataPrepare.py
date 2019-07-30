# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:22:48 2019

@author: Raytine
"""
import os
import cv2
import numpy as np
import xml.dom.minidom
from tqdm import tqdm

def he_to_binary_mask(xml_path, img_path, label_path):
    # analysis xml
    dom = xml.dom.minidom.parse(xml_path)
    Regions = dom.getElementsByTagName('Region')
    xy_all = []
    for regioni in range(len(Regions)):
        Region = Regions.item(regioni)
        verticies = Region.getElementsByTagName('Vertex')
        xy = []
        for vertexi in range(len(verticies)):
            x = float(verticies.item(vertexi).getAttribute('X'))
            y = float(verticies.item(vertexi).getAttribute('Y'))
            xy.append([x,y])
        xy_all.append(xy)
    # analysis img
    im_info = cv2.imread(img_path)
    nrow=im_info.shape[0]
    ncol=im_info.shape[1]
    binary_mask=np.zeros([nrow,ncol], dtype="uint8")
    for zz in range(len(xy_all)):
        x_cor = np.array(xy_all[zz])[:,0]
        x_cor = x_cor.reshape(x_cor.shape+(1,))
        y_cor = np.array(xy_all[zz])[:,1]
        y_cor = y_cor.reshape(y_cor.shape+(1,))
        cor_xy = np.hstack((x_cor, y_cor))
        cv2.polylines(binary_mask, np.int32([cor_xy]), 1, 1)
        cv2.fillPoly(binary_mask, np.int32([cor_xy]), 1)
    cv2.imwrite(label_path, binary_mask*255)
#    cv2.imwrite(img_path.replace('tif','png'), im_info)
    
if __name__ == '__main__':
    filepath = '.\\data\\MoNuSeg\\train\\image'
    pathDir =  os.listdir(filepath)
    for allDir in tqdm(pathDir):
        img_path = os.path.join(filepath, allDir)
        xml_path = img_path.replace('tif','xml').replace('image','Annotation')
        label_path = img_path.replace('tif','png').replace('image','label')
        he_to_binary_mask(xml_path, img_path, label_path)
