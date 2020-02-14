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
num_cluster = 8

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

def MSE(box,centers):
    mns = centers - box
    return np.sqrt(mns[:,0]**2 + mns[:,1]**2) 

def draw_cluster_result(sample_array,belonging,centers):
    sample_num = sample_array.shape[0]
    colored_sample_0 = np.empty(shape=[0,2])
    colored_sample_1 = np.empty(shape=[0,2])
    colored_sample_2 = np.empty(shape=[0,2])
    colored_sample_3 = np.empty(shape=[0,2])
    colored_sample_4 = np.empty(shape=[0,2])
    colored_sample_5 = np.empty(shape=[0,2])
    colored_sample_6 = np.empty(shape=[0,2])
    colored_sample_7 = np.empty(shape=[0,2])
    colored_sample_8 = np.empty(shape=[0,2])
    for i in range(sample_num):
        if belonging[i] == 0:
            colored_sample_0 = np.append(colored_sample_0,[sample_array[i]],axis=0)
        elif belonging[i] == 1:
            colored_sample_1 = np.append(colored_sample_1,[sample_array[i]],axis=0)
        elif belonging[i] == 2: 
            colored_sample_2 = np.append(colored_sample_2,[sample_array[i]],axis=0)
        elif belonging[i] == 3: 
            colored_sample_3 = np.append(colored_sample_3,[sample_array[i]],axis=0)
        elif belonging[i] == 4: 
            colored_sample_4 = np.append(colored_sample_4,[sample_array[i]],axis=0)
        elif belonging[i] == 5: 
            colored_sample_5 = np.append(colored_sample_5,[sample_array[i]],axis=0)
        elif belonging[i] == 6: 
            colored_sample_6 = np.append(colored_sample_6,[sample_array[i]],axis=0)
        elif belonging[i] == 7: 
            colored_sample_7 = np.append(colored_sample_7,[sample_array[i]],axis=0)
        else: 
            colored_sample_8 = np.append(colored_sample_8,[sample_array[i]],axis=0)
    plt.plot(colored_sample_0[:,0],colored_sample_0[:,1],'o',markersize=1,color='b')
    plt.plot(colored_sample_1[:,0],colored_sample_1[:,1],'o',markersize=1,color='g')
    plt.plot(colored_sample_2[:,0],colored_sample_2[:,1],'o',markersize=1,color='r')
    plt.plot(colored_sample_3[:,0],colored_sample_3[:,1],'o',markersize=1,color='c')
    plt.plot(colored_sample_4[:,0],colored_sample_4[:,1],'o',markersize=1,color='m')
    plt.plot(colored_sample_5[:,0],colored_sample_5[:,1],'o',markersize=1,color='y')
    plt.plot(colored_sample_6[:,0],colored_sample_6[:,1],'o',markersize=1,color='k')
    plt.plot(colored_sample_7[:,0],colored_sample_7[:,1],'o',markersize=1,color=(0.1,0.7,0.5))
    plt.plot(colored_sample_8[:,0],colored_sample_8[:,1],'o',markersize=1,color=(0.4,0.7,0.1))
    plt.plot(centers[:,0],centers[:,1],'o',color='m')
    plt.show()

def kmeans(sample_array,centers):
    sample_num = sample_array.shape[0]
    last_belonging = np.ones(shape=[sample_num])     
    iter_num = 0
    
    while(1):
        distance = []
        iter_num += 1
        for i in range(sample_num):
            d = 1 - IOU(sample_array[i],centers)
            #d = MSE(sample_array[i],centers)
            distance.append(d)
        distance = np.array(distance)
        belonging = np.argmin(distance,axis=1)
        if(belonging == last_belonging).all():
            print("Finish clustering after %d round,centers are"%iter_num)
            print(centers)

            #Sum of error,instead of SSE for distance(1 - IOU) in this project is less than 1,
            #if we use SSE,the longer distance will become shorter
            se = 0
            for i in range(sample_num):
                se += distance[i,belonging[i]]
            print("Everage of error is " + str(se/sample_num))

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

    return belonging,centers       
        
def main():
    indexes = []
    boxes = get_sample(xml_dir)
    for i in range(num_cluster):
        indexes.append(random.randrange(boxes.shape[0]))
    init_centers = boxes[indexes]
    belonging,centers = kmeans(boxes,init_centers)
    draw_cluster_result(boxes,belonging,centers)
    #print(init_center)


if __name__ == "__main__":
    main()

