#
#########################################################################
# Author:smallbean 
# Created Time: 12/02/2010
# File Name: anchor_cluster.py
# Description:
#            Use kmeans method to cluster anchors in yolov3 
#########################################################################

import numpy
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from lxml.etree import Element,SubElement,tostring,ElementTree

#Annotation xml file folder
xml_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/label_training/label_2_xml/'
num_cluster = 9

def get_sample_from_one_file(sample_list,file):
    tree = ElementTree()    
    tree.parse(xml_dir + file)
    root = tree.getroot()
    while(1):
        obj = root.find('object')
        if obj is None:
            break
        bb = obj.find('bndbox')
        xmin = bb.find('xmin').text
        ymin = bb.find('ymin').text
        xmax = bb.find('xmax').text
        ymax = bb.find('ymax').text

        w = int(xmax) - int(xmin)
        h = int(ymax) - int(ymin)
        sample_list.append([w,h])
        root.remove(obj)

#Collect sample in every xml files
def get_sample(dir):
    all_sample = []
    for _,_,files in os.walk(dir): 
        for file in files:
            get_sample_from_one_file(all_sample,file)
    return np.array(all_sample)
    #print(all_sample)


def IOU(box,centers):
    ious = []
    w,h = box
    for tbox in centers:    
        t_w,t_h = tbox
        h_min = min(h,t_h)
        w_min = min(w,t_w)
        iou = w_min*h_min/(w*h + t_w*t_h - w_min*h_min)
        ious.append(iou)
    return np.array(ious)


def kmeans(sample_array,centers):
    sample_num = sample_array.shape[0]
    last_belonging = np.ones(sample_num) * (0)    
    iter_num = 0
    
    while(1):
        distance = []
        iter_num += 1
        for i in range(sample_num):
            d = 1 - IOU(sample_array[i],centers)
            distance.append(d)
        distance = np.array(distance)
        belonging = np.argmin(distance,axis=1)
        #print("Now centers are")
        #print(centers)
        if(belonging == last_belonging).all():
            print("Finish clustering after %d round,centers are"%iter_num)
            print(centers)

            #Sum of error,instead of SSE for distance(1 - IOU) in this project is less than 1,
            #if we use SSE,the longer distance will become shorter
            se = 0
            for i in range(sample_num):
                se += distance[i,belonging[i]]
            print("Everage of error is " + str(se/sample_num))
            plt.plot(sample_array[:,0],sample_array[:,1],'o',markersize=1)
            plt.plot(centers[:,0],centers[:,1],'o',color=(0.8,0.,0.))
            plt.show()

            break

        last_belonging = belonging        
        #calculate new centers
        centers_value = np.zeros(((centers.shape[0]),2),np.float)
        centers_sample_num = np.zeros((centers.shape[0]),np.int)

        for i in range(sample_num):
            centers_value[belonging[i]] += sample_array[i]
            centers_sample_num[belonging[i]] +=1
        for i in range(centers.shape[0]):
            centers[i] = centers_value[i]/centers_sample_num[i]

        
def main():
    indexes = []
    boxes = get_sample(xml_dir)
    for i in range(num_cluster):
        indexes.append(random.randrange(boxes.shape[0]))
    init_centers = boxes[indexes]
    kmeans(boxes,init_centers)
    #print(init_center)


if __name__ == "__main__":
    main()

