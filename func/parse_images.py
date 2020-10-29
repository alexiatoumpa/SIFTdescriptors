import __init__ as init

# Export output
from write_data import WriteInFile
from sift import SIFTdesc

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import math
import time
import datetime 
import itertools
from itertools import combinations
import copy
import csv
from scipy import ndimage
from PIL import Image

def ProcessImages():
    # Parse the directory
    PATH = '/home/scat/Datasets/CAD-120/images_objects/images/'

    frame_init = 1

    for sub in os.listdir(PATH): 
        subject = PATH + sub + '/'
        for act in os.listdir(subject):
            activity = subject + act + '/'
            for tas in os.listdir(activity):
                task = activity + tas + '/'
                print(task)
                numobjects = len(os.listdir(task))
                for obj in os.listdir(task):
                    objects = task + obj + '/'
                    
                    lendir = len(os.listdir(objects))/2

                    for fr in range(frame_init,lendir+1,1):
                        imgname = objects + 'obj_'+ obj[-1:] + '_RGB_' + str(fr) + '.png'
                        img = cv2.imread(imgname)
                        # depth
                        dimgname = objects + 'obj_'+ obj[-1:] + '_Depth_' + str(fr) + '.png'
                        dimg = cv2.imread(dimgname,0)
                        rgbdimg = cv2.cvtColor(dimg, cv2.COLOR_GRAY2RGB)

                        #test_image = imgname
                        #oriImg = cv2.imread(test_image) # B,G,R order
                        #boxsize = 480
                        #scale = boxsize / (oriImg.shape[0] * 1.0)
                        #img = cv2.resize(oriImg, (0,0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)


                        # ------------------- For the Objects -------------------#
                        hole_max, hole_min, data_max, data_min = 0, 0, 0, 0
                        holecolor = ['green', 'blue']
                        objTHRES = [[0,0] for i in range(numobjects)] # [min,max] : saving min and max thresholds of depth distribution for all objects in the scene
                        
                        for numo in range(0,numobjects): # numo: is also the ID of the objects as numobjects defines the total number of objects.
                            this_object_id = numo + 1
                            _y = len(img)
                            _x = len(img[0])

            
                            # Keep in depth_info all the depth information of the objects' bounding box.
                            depth_info = []
                            for x in range(int(_x)):
                                for y in range(int(_y)):
                                    datapoint = dimg[y][x]
                                    if datapoint!= 0 and datapoint!=255: # NOISE: if 0 or 255
                                        depth_info.append(datapoint)

                            # Check if the depth information is valuable. If not then discard the object as it is out of the range of the sensor.
                            if len(depth_info) != 0:
                                depth_info = sorted(depth_info)

                                # Finding the group where the object is most likely to exist considering the derivative of the depth distribution.   
                                init.sensitivity = 10 # 10 is good
                                #init.h = 1 # secant line 
                                # Slope of the secant line: Q(h) = (f[i] - f[i-h]) / h                            
                                derivative_checkpoints = [i for i in range(1, len(depth_info)) if (((depth_info[i] - depth_info[i-init.h])/init.h) > init.sensitivity)]
                                derivative_checkpoints.insert(0,0)
                                derivative_checkpoints.append(len(depth_info))
                                obj_group_s, obj_group_f = 0,0
                                max_difference = 0 # find maximum amount of depth pixels corresponding to the object being detected
                                # Find the longest sequence of sorted depth infromation, according to the derivative peaks.
                                for i in range(1,len(derivative_checkpoints)):
                                    difference_groups = derivative_checkpoints[i] - derivative_checkpoints[i-1]
                                    if max_difference < difference_groups:
                                        max_difference = difference_groups
                                        obj_group_s, obj_group_f = derivative_checkpoints[i-1], derivative_checkpoints[i]
                                    
                                # Coarse-graining the depth list according to the derivative and the biggest group of pixels
                                depth_list = copy.deepcopy(depth_info[obj_group_s:obj_group_f])
                                pixel_list = range(obj_group_f - obj_group_s)

                                # depth_list: the list of the depth information of the object for the corresponding bounding box.
                                dmean = np.mean(depth_list)
                                sigma = np.std(depth_list)
                                thres_min = dmean - sigma
                                thres_max = dmean + sigma


                                # Need to condition this , when the thres_area is big enough to be considered as an object which might contain a hole.
                                    
                                # Because the distance of the camera is not linear this needs to be changed to something which is more robust
                                # Because as it is now, this needs to change depending on the distance of the camera.
                                thres_area = thres_max - thres_min
                                init.how_many_max_sections = 3 # 2 is GOOD # 3
                                init.divider = 1.0
                                    
                                if thres_area > init.MAX_thres_area: # CONCAVE OBJECT
                                    #objTYPE[fr][numo] = 'concave'
                                    init.divider = 5.0 # 4.0 is GOOD # PARAMETER
                                    section = thres_area / init.divider

                                    minDeepdatapoint = thres_max - section * init.how_many_max_sections
                                    objTHRES[numo][0] = minDeepdatapoint # hole_min
                                    objTHRES[numo][1] = thres_max # hole_max
                                    
                                    for x in range(int(_x)):
                                        for y in range(int(_y)):
                                            datapoint = dimg[y][x]
                                            #if datapoint>= minDeepdatapoint and datapoint <= thres_max: # BLUE: the furthest away object area (concave curve)                                      
                                            #    cv2.circle(img,(x,y),1,[255,0,0],-1)    
                                            #elif datapoint >= minDeepdatapoint-section and datapoint <= minDeepdatapoint:
                                            #    cv2.circle(img,(x,y),1,[0,255,255],-1) # YELLOW: the middle object area
                                            #elif datapoint >= thres_min and datapoint <= minDeepdatapoint-section:
                                            #    cv2.circle(img,(x,y),1,[150,150,40],-1) # CYAN: the closest object area
                                            
                                            if datapoint > thres_max:
                                                cv2.circle(img,(x,y),1,[0,255,0],-1) # GREEN: the furthest away
                                            #elif datapoint < thres_min: # datapoint < thres_min
                                            #    cv2.circle(img,(x,y),1,[0,255,0],-1) # RED: the closest or noise
                                            
                                           

                                else: # CONVEX OBJECT
                                    #objTYPE[fr][numo] = 'convex'
                                    objTHRES[numo][0] = thres_min # data_min
                                    objTHRES[numo][1] = thres_max # data_max
                                    for x in range(int(_x)):
                                        for y in range(int(_y)):
                                            datapoint = dimg[y][x]
                                            if datapoint < depth_list[0] or datapoint > depth_list[len(depth_list)-1]:
                                                cv2.circle(img,(x,y),1,[0,255,0],-1)

                        # ------------------- End of Objects -------------------#
                        img, desc = SIFTdesc(img, select_kp=True)
			#norm_desc = SIFTnormalize(desc)
			
			img2 = Image.fromarray(img, 'RGB')
                        cv2.imshow('CAD-120: QSR for bounding boxes',img)
			img2.show() 

                        k = cv2.waitKey(30) & 0xff
                        #'''
                        if k == 27:
                            #cv2.imwrite(os.path.join('/home/toumpa/Dropbox/Canny/saved_figs' , 'qsr_0.jpg'), img)
                            #cv2.waitKey(0)
                            break
                        '''

                        # Stopping in every frame
                        while (k!=27):
                            k = cv2.waitKey(30) & 0xff
                            if k == 27:
                                break
                        '''


                    print('All Done')
                    cv2.destroyAllWindows()
