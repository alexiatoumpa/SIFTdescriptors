#
# Copyright (C) 2019 University of Leeds
# Author: Alexia Toumpa
# email: scat@leeds.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
#

import __init__ as init

# Export output
from write_data import WriteInFile

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import math
import time
#import pandas as pd
import networkx as nx
import datetime 
import itertools
from itertools import combinations
import copy
import csv
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage



def ProcessImages():

	unique_graph_name = 'aa'
	detect3D = False
	show_qsr= False
	show_image = False


    # Parse the directory
    WHICH_SET = '' # {training, validation, test}
    DIR = 'TrainQSR-FRCNN'

    PATH = '/home/scat/Datasets/CAD-120/' + DIR + '/'+ WHICH_SET + '/images/'
    ANNOTATIONS = '/home/scat/Datasets/CAD-120/enhanced_annotations/'
    DENSEPOSE = '/home/scat/Datasets/CAD-120/images_densepose/'

    init.use_rcnn = False
    
    if init.use_rcnn:
        ANNOT_RCNN = '/home/scat/Datasets/CAD-120/annotations_rcnn/'
    else:
        OBJECT_LABELS = '/home/scat/Datasets/CAD-120/' + DIR + '/' + WHICH_SET + '/files/object_labels'
        VIDEO_OBJ_LIST = '/home/scat/Datasets/CAD-120/' + DIR + '/' + WHICH_SET + '/files/exported_listdir'


   


    frame_init = 1#1

    # Visualizing the depth distribution
    Visualize_plot = False

    deletelonely = True
    #detect3D = True
    conceptual_PO = True
    median_filtering = True
    #unique_graph_name = 'spec_val'

    ### For the graphs plotting (not the activity graphs)
    # Single joints
    Jhead = []
    Jhands = []
    Jbody = []
    Jfeet = []
    # Joint joints
    J_head_hands = []
    J_hands_feet = []

    # For the satio-temporal extracted features
    F_list = []
    spatial_F_list = []
    unique_spatial_list = []
    spatial_Fobjs_list = []
    # For saving the clustering data
    video_name_list = []
    video_task_list = []
    video_objs_list = []


    if save_affordances:
        fa = open(gt_affordance_file, "a+")

    if init.write_in_file:
        f=open(init.filename, "a+")

    for sub in os.listdir(PATH): 
        #if True:
        #sub = 'Subject3_rgbd_images'
        subject = PATH + sub + '/'
        for act in os.listdir(subject):
            #if True:
            #act = 'taking_food'
            activity = subject + act + '/'
            for tas in os.listdir(activity):
                #if True:
                #tas = '0510144139'
                task = activity + tas + '/'
                print(task)
                #sub = 'Subject3_rgbd_images' # FORCED
                #act = 'taking_food' # FORCED
                #tas = '0510144139' # FORCED
                #task = '/home/scat/Datasets/CAD-120/TrainQSR-FRCNN/training/images/Subject3_rgbd_images/taking_food/0510144139/'
                pathskel = ANNOTATIONS + sub[0:8] + '_annotations/' + act + '/' + tas + '_cpm.txt'
                dense_path = DENSEPOSE + sub + '/' + act + '/' + tas + '/'

                if init.use_rcnn:
                    pathobj = ANNOT_RCNN + sub + '/' + act + '/'
                
                lendir = len(os.listdir(task))/2
                

                
                
                

                if init.write_in_file:
                    f.write("%s\r\n" % task)
                
                file = open(pathskel, 'r')
                lines_skel = file.readlines()
                numlines = len(lines_skel)
                
                # Initialize orientation(ORI) and position(P) matrices
                init.P = np.zeros(((numlines,14,1,2))) # 1x2 array for position of all joints(14) for all frames(numlines): numlinesx14x(1x2)
                readSkeletonsCPM(lines_skel, init.P)
                
                if init.use_rcnn:
                    numobjects = sum(1 for i in os.listdir(pathobj) if tas in i)
                else:
                    # Check the number of objects in the scene
                    check_obj = tas + '_obj'
                    check_path = ANNOTATIONS + sub[0:8] + '_annotations/' + act + '/'
                    numobjects = checkNumObjects(check_obj, check_path)

                # Previous Object Location
                PrevObj = np.zeros(((numlines+1,numobjects,4)))
                PrevHands = np.zeros(((numlines+1,2,4))) # same as PrevObjs but for the two hands
                ObjID = [[0 for i in range(numobjects)] for j in range(numlines+1)] #[[0] * numobjects] * (numlines+1)
                objTYPE = [[0 for i in range(numobjects)] for j in range(numlines+1)] # [[] for i in range(numobjects)]
                
                Lqsr = [[] for i in range(numobjects)] #[[]] * numobjects # used in append(), so no need for [0]
                Lqsr_hands = [ [ [ [] for i in range(2)] for h in range(2)] for o in range(numobjects)] # [ [ [[o0],[h0]],[[o0],[h1]] ]
                objlist_pervideo = [[] for i in range(numobjects)] #[[]] * numobjects
                prev_pos = np.zeros((numobjects,4))
                depth_object = np.zeros(numobjects)

                # Initializing
                init.UL = np.zeros(((numobjects,numlines,1,2))) # 1x2 matrix for upper left corner of bounding box: numlinesx(1x2)
                init.LR = np.zeros(((numobjects,numlines,1,2))) # 1x2 matrix for lower right corner of bounding box: numlinesx(1x2)
                #init.T = np.zeros((((numobjects,numlines,6)))) # 6x1 array for transformation matrix matching the SIFT features to the previous

                # Keep the nameID of the objects for visualization
                names = []

                if init.use_rcnn: 
                    nmo = 0
                    for objfile in os.listdir(pathobj):
                        if tas in objfile:
                            pathobj_obj = pathobj + objfile

                            splitted = objfile.split('_')
                            name_object = splitted[1][:-4][3:]
                            names.append(name_object)
                            #print(nmo, name_object)

                            file = open(pathobj_obj, 'r')
                            frame_lines = file.readlines()
                            if len(frame_lines)>numlines:
                                frame_lines = frame_lines[:-1]  
                            readObjectsRCNN(frame_lines, nmo, init.UL, init.LR) 
                            nmo += 1
                else:
                    for nmo in range(0, numobjects):
                        nmo2 = nmo+1
                        pathobj = ANNOTATIONS + sub[0:8] + '_annotations/' + act + '/' + check_obj + str(nmo2) + '.txt'
                        names.append(str(nmo2))
                        file = open(pathobj, 'r')
                        frame_lines = file.readlines()
                        if len(frame_lines)>numlines:
                            frame_lines = frame_lines[:-1]
                        readObjects(frame_lines, nmo, init.UL, init.LR)
                        
                obj_relations_prev = 0
                print_interactions = True
                print_objs_inter = True
                print_obj_list = []
                print_int_list = []
                include_more = False
                depth_list = []
                depth_of_1_object = True

                # Used for filtering out single appearances of DC between C-type RCC relations.
                obj_combinations = list(itertools.combinations(range(numobjects),2))
                num_obj_combinations = len(obj_combinations)
                init.ITISC = [0 for i in range(num_obj_combinations)]
                init.ITISDC = [0 for i in range(num_obj_combinations)]
                dict_combinations = {}
                dnum = 0
                for d in range(num_obj_combinations):
                    dict_combinations[str(obj_combinations[d][0]) + ',' + str(obj_combinations[d][1])] = dnum
                    dnum += 1

                
                # For all the combinations create a list of interactions and of the correlated objects.
                Linteractions = [[[],[]] for i in range(len(dict_combinations))]
                objlist_interactions = [[] for i in range(len(dict_combinations))]



                for fr in range(frame_init,lendir+1,1):
                    imgname = task + 'RGB_' + str(fr) + '.png'
                    img = cv2.imread(imgname)
                    # depth
                    dimgname = task + 'Depth_' + str(fr) + '.png'
                    dimg = cv2.imread(dimgname,0)
                    rgbdimg = cv2.cvtColor(dimg, cv2.COLOR_GRAY2RGB)

                    test_image = imgname
                    oriImg = cv2.imread(test_image) # B,G,R order
                    boxsize = 480
                    scale = boxsize / (oriImg.shape[0] * 1.0)
                    #imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    img = cv2.resize(oriImg, (0,0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)

                    dense_imgname = dense_path + 'RGB_' + str(fr) + '_IUV.png'
                    denseimg = cv2.imread(dense_imgname)

                    imgobj = cv2.imread(dimgname)


                   


                    # ------------------- For the Objects -------------------#
                    Rnumo = 0
                    count_spatial_nodes = 0
                    hole_max, hole_min, data_max, data_min = 0, 0, 0, 0
                    #fig.clear() # - Problem with the 3D models: clears them out
                    if Visualize_plot:
                        ax1.clear()
                        ax2.clear()
                        ax3.clear()
                        ax4.clear()
                    holeobj = 0
                    holecolor = ['green', 'blue']
                    #meancolo = ['r-', 'k-']
                    #linecolor = ['b.', 'k.']
                    #objTYPE = [[] for i in range(numobjects)]
                    objTHRES = [[0,0] for i in range(numobjects)] # [min,max] : saving min and max thresholds of depth distribution for all objects in the scene
                    for numo in range(0,numobjects): # numo: is also the ID of the objects as numobjects defines the total number of objects.
                        this_object_id = numo + 1
        
                        # get object's coordinates for previous frame
                        x_ul, y_ul, x_lr, y_lr = visualizeObject(fr-1, numo)

                        if not init.use_rcnn:
                            # Find object type
                            objpath = sub[0:8] + '_rgbd_images/' + act + '/' + tas + '/object_' + str(numo+1) + '/'
                            file = open(VIDEO_OBJ_LIST, 'r')
                            row = file.readlines()
                            lc = 0
                            for line in row:  
                                if (line[:-1] == objpath):
                                    #print('Found the object.')
                                    break
                                else:
                                    lc += 1

                            fileo = open(OBJECT_LABELS, 'r')
                            rowo = fileo.readlines()
                            lo = 0
                            for lineo in rowo:
                                if (lo < lc):
                                    lo += 1
                                else:
                                    objtype = lineo
                                    break

                        # Deal with the (0,0,0,0) annotations of the objects
                        if x_ul > 640:
                            x_ul = 640
                        if x_lr > 640:
                            x_lr = 640
                        if y_ul > 480:
                            y_ul = 480
                        if y_lr > 480:
                            y_lr = 480
                        if x_ul < 0:
                            x_ul = 0
                        if x_lr < 0:
                            x_lr = 0
                        if y_ul < 0:
                            y_ul = 0
                        if y_lr < 0:
                            y_lr = 0

                        #if (x_ul > 0 and y_ul > 0 and x_lr > 0 and y_lr > 0 and x_ul < 640 and x_lr < 640 and y_ul < 480 and y_lr < 480) and \
                        #    (not (hx >= x_ul and hx <= x_lr and hy >= y_ul and hy <= y_lr) and not (nx >= x_ul and nx <= x_lr and ny >= y_ul and ny <= y_lr)):
                        if (not (x_ul == 0 and y_ul == 0 and x_lr == 0 and y_lr == 0)) and \
                            (not (hx >= x_ul and hx <= x_lr and hy >= y_ul and hy <= y_lr) and not (nx >= x_ul and nx <= x_lr and ny >= y_ul and ny <= y_lr)):
                        
                            # Visualize the bounding box
                            cv2.line(img, (int(x_ul),int(y_ul)),(int(x_lr),int(y_ul)), (255,255,255), 4)
                            cv2.line(img, (int(x_ul),int(y_ul)),(int(x_ul),int(y_lr)), (255,255,255), 4)
                            cv2.line(img, (int(x_lr),int(y_lr)),(int(x_lr),int(y_ul)), (255,255,255), 4)
                            cv2.line(img, (int(x_lr),int(y_lr)),(int(x_ul),int(y_lr)), (255,255,255), 4)

                            cv2.line(rgbdimg, (int(x_ul),int(y_ul)),(int(x_lr),int(y_ul)), (255,255,255), 1)
                            cv2.line(rgbdimg, (int(x_ul),int(y_ul)),(int(x_ul),int(y_lr)), (255,255,255), 1)
                            cv2.line(rgbdimg, (int(x_lr),int(y_lr)),(int(x_lr),int(y_ul)), (255,255,255), 1)
                            cv2.line(rgbdimg, (int(x_lr),int(y_lr)),(int(x_ul),int(y_lr)), (255,255,255), 1)

                            # Keep in depth_info all the depth information of the objects' bounding box.
                            depth_info = []
                            for x in range(int(x_ul), int(x_lr)):
                                for y in range(int(y_ul), int(y_lr)):
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

                                # Make 3D objects from bounding boxes
                                if Visualize_plot:
                                    if numo == 0:
                                        ax3d = ax3
                                        ax = ax1
                                    else:
                                        ax3d =ax4
                                        ax = ax2

                                    area_ofBB = abs(int(x_lr) - int(x_ul)) * abs(int(y_lr) - int(y_ul))
                                    X = np.zeros(area_ofBB)
                                    Y = np.zeros(area_ofBB)
                                    Z = np.zeros(area_ofBB)
                                    s = 0
                                    for x in range(int(x_ul), int(x_lr)):
                                        for y in range(int(y_ul), int(y_lr)):
                                            Z[s] = dimg[y][x]
                                            X[s] = x
                                            Y[s] = y
                                            s += 1
                                    Zl = Z
                                    Z1 = Zl[Zl < thres_min]
                                    Z3 = Zl[Zl > thres_max]
                                    Z1 = np.asarray(Z1)
                                    Z3 = np.asarray(Z3)
                                    Xl = X
                                    X1 = Xl[Zl < thres_min]
                                    X3 = Xl[Zl > thres_max]
                                    X1 = np.asarray(X1)
                                    X3 = np.asarray(X3)
                                    Yl = Y
                                    Y1 = Yl[Zl < thres_min]
                                    Y3 = Yl[Zl > thres_max]
                                    Y1 = np.asarray(Y1)
                                    Y3 = np.asarray(Y3)
                                    # plot(x,y,z, color)
                                    ax3d.plot(X, Z, Y, 'r.')
                                    ax3d.set_xlabel('x coordinates')
                                    ax3d.set_ylabel('depth')
                                    ax3d.set_zlabel('y coordinates')
                                    ax3d.plot(X1, Z1, Y1, 'k.') # points very close
                                    ax3d.plot(X3, Z3, Y3, 'b.') # points very far
                                    ax.set_ylim([0,60])
                                    ax.set_xlim([0,len(depth_list)+50])
                                    ax.set_ylabel('Depth')
                                    ax.set_xlabel('Pixels')
                                    ax.plot([0, max(pixel_list)], [dmean, dmean], 'r-', lw = 1)
                                    ax.plot(pixel_list, depth_list,'k.')
                                    plt.pause(0.5)
                                # Need to condition this , when the thres_area is big enough to be considered as an object which might contain a hole.
                                
                                # Because the distance of the camera is not linear this needs to be changed to something which is more robust
                                # Because as it is now, this needs to change depending on the distance of the camera.
                                thres_area = thres_max - thres_min
                                init.how_many_max_sections = 3 # 2 is GOOD # 3
                                init.divider = 1.0
                                
                                if thres_area > init.MAX_thres_area: # CONCAVE OBJECT
                                    objTYPE[fr][numo] = 'concave'
                                    init.divider = 5.0 # 4.0 is GOOD # PARAMETER
                                    section = thres_area / init.divider

                                    # Show plot of the sorted distribution of the depth over the image in 2D
                                    #alpha_values = 1.0 / init.divider
                                    #startp = thres_min
                                    #starta = 0.0
                                    #fillcol = holecolor[holeobj]
                                    #for i in range(int(init.divider)):
                                    #    ax.fill_between([0, max(pixel_list)], startp, startp + section, facecolor = fillcol, alpha = starta + alpha_values)
                                    #    startp += section
                                    #    starta += alpha_values

                                    # Show where the derivative makes a peak
                                    #for i in derivative_checkpoints[1:-1]:
                                    #    ax.plot([i,i], [0, max(depth_list)], 'g-')

                                    minDeepdatapoint = thres_max - section * init.how_many_max_sections
                                    objTHRES[numo][0] = minDeepdatapoint # hole_min
                                    objTHRES[numo][1] = thres_max # hole_max
                                    if Visualize_plot:
                                        # Visualizing zones
                                        bluezone_max = thres_max
                                        bluezone_min = minDeepdatapoint
                                        yellowzone_max = minDeepdatapoint
                                        yellowzone_min = minDeepdatapoint - section
                                        cyanzone_max = minDeepdatapoint - section
                                        cyanzone_min = minDeepdatapoint - section * 2 # 3
                                        ax.fill_between([0, max(pixel_list)], bluezone_min, bluezone_max, facecolor = 'blue', alpha = 0.4)
                                        ax.fill_between([0, max(pixel_list)], yellowzone_min, yellowzone_max, facecolor = 'yellow')
                                        ax.fill_between([0, max(pixel_list)], cyanzone_min, cyanzone_max, facecolor = 'cyan', alpha = 0.4)
                                        ax.fill_between([0, max(pixel_list)], np.max(depth_list), thres_max, facecolor = 'black', alpha = 0.2)
                                        # Area outlide the limits of the standard deviation is considered the gray zone.
                                        if np.min(depth_list) < thres_min:
                                            ax.fill_between([0, max(pixel_list)], np.min(depth_list), thres_min, facecolor = 'black', alpha = 0.4)
                                        plt.pause(0.5)
                                    holeobj += 1
                                    for x in range(int(x_ul), int(x_lr)):
                                        for y in range(int(y_ul), int(y_lr)):
                                            datapoint = dimg[y][x]
                                            if datapoint>= minDeepdatapoint and datapoint <= thres_max: # BLUE: the furthest away object area (concave curve)                                      
                                                cv2.circle(img,(x,y),1,[255,0,0],-1)
                                                
                                            elif datapoint >= minDeepdatapoint-section and datapoint <= minDeepdatapoint:
                                                cv2.circle(img,(x,y),1,[0,255,255],-1) # YELLOW: the middle object area
                                            elif datapoint >= thres_min and datapoint <= minDeepdatapoint-section:
                                                cv2.circle(img,(x,y),1,[150,150,40],-1) # CYAN: the closest object area
                                            elif datapoint > thres_max:
                                                cv2.circle(img,(x,y),1,[0,0,0],-1) # BLACK: the furthest away
                                            else: # datapoint < thres_min
                                                cv2.circle(img,(x,y),1,[0,0,255],-1) # RED: the closest or noise
                                    """
                                    Find which of the concave objects are actual concave and which of them are convex long surfaces.
                                    To differentiate between these two we search for the concave curve by:
                                    - finding the contour of the BLUE area
                                    - define the hierarchy of the contours and pick the inner child if it significant enough in terms of area occupying.
                                    """
                                    # crop the bounding box of the object
                                    crop_obj = img[int(y_ul):int(y_lr), int(x_ul):int(x_lr)]
                                    # compute are of the bounding box of the object
                                    obj_area = abs(y_lr-y_ul)*abs(x_lr-x_ul)
                                    # convert to gay scale
                                    crop_gray = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2GRAY)
                                    # binarize the information
                                    threshold_value = crop_gray.min() #1
                                    ret, thresh = cv2.threshold(crop_gray, threshold_value, 255, 0)
                                    # find hierarchy of contours 
                                    contour_obj, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    # Draw contours
                                    hull = []
                                    for contour in contours:
                                        hull.append(cv2.convexHull(contour, False))
                                    drawing = np.zeros((thresh.shape[0], thresh.shape[1]))
                                    color_contours = (0,255,0) # green
                                    color = (255,0,0) # blue
                                    for i in range(len(contours)):
                                        #cv2.drawContours(drawing, contours, i , color_contours, 1, 8, hierarchy)
                                        cv2.drawContours(drawing, hull, i, color, 1, 8)
                                    
                                    # compute area of the concave area
                                    multiplier_value = 0.05
                                    areas = []
                                    for c in contours:
                                        areas.append(cv2.contourArea(c))
                                    #if names[numo] == '1':
                                    #    print('this is 1')
                                    #    print(type(thresh))
                                    #    thres11 = copy.deepcopy(thresh)
                                    #    crop_gray11 = copy.deepcopy(crop_gray)
                                    #if names[numo] == '6':
                                    #    print('this is 6')
                                    #thres11 = copy.deepcopy(thresh)
                                    #crop_gray11 = copy.deepcopy(crop_gray)
                                    #print(areas)
                                    #print(hierarchy)
                                    #print(multiplier_value*obj_area, obj_area)

                                    # Find if the object in concave or a convex surface by looking if it has any parent, no children, and exceeds the area threshold.
                                    is_this_convex = True
                                    for i in range(len(areas)):
                                        hierarchy_info = hierarchy[0][i]
                                        parent = hierarchy_info[3]
                                        child = hierarchy_info[2]
                                        area = areas[i]
                                        if parent!=-1 and child==-1 and area>=(multiplier_value*obj_area): # this is a concave object : parent!=-1 and 
                                            #print(areas)
                                            is_this_convex = False
                                    
                                    for f in range(fr):
                                        if objTYPE[f][numo] == 'concave':
                                            #print('Remember concaveness.')
                                            is_this_convex = False
                                            break
                                    
                                    if is_this_convex:
                                        # this is a convex object
                                        objTYPE[fr][numo] = 'convex_surface'
                                        objTHRES[numo][0] = thres_min # data_min
                                        objTHRES[numo][1] = thres_max # data_max
                                        for x in range(int(x_ul), int(x_lr)):
                                            for y in range(int(y_ul), int(y_lr)):
                                                datapoint = dimg[y][x]
                                                if datapoint >= thres_min and datapoint <= thres_max:
                                                    #if datapoint>= depth_list[0] and datapoint <= depth_list[len(depth_list)-1]:
                                                    cv2.circle(img,(x,y),1,[0,0,0],-1)
                                                else:
                                                    cv2.circle(img,(x,y),1,[255,255,255],-1)
                                        
                                        ''' # TODO: MASK OF THE convex_surface OBJECT
                                        # convert to gay scale
                                        crop_gray = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2GRAY)
                                        # binarize the information
                                        threshold_value = 1
                                        ret, thresh = cv2.threshold(crop_gray, threshold_value, 255, 0)
                                        # find hierarchy of contours 
                                        contour_obj, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                        # Draw contours
                                        hull = []
                                        for contour in contours:
                                            hull.append(cv2.convexHull(contour, False))
                                        drawing = np.zeros((thresh.shape[0], thresh.shape[1]))
                                        color = (255,0,0) # blue
                                        for i in range(len(contours)):
                                            cv2.drawContours(drawing, hull, i, color, 1, 8)
                                        '''
                                        

                                else: # CONVEX OBJECT
                                    objTYPE[fr][numo] = 'convex'
                                    objTHRES[numo][0] = thres_min # data_min
                                    objTHRES[numo][1] = thres_max # data_max
                                    for x in range(int(x_ul), int(x_lr)):
                                        for y in range(int(y_ul), int(y_lr)):
                                            datapoint = dimg[y][x]
                                            if datapoint>= depth_list[0] and datapoint <= depth_list[len(depth_list)-1]:
                                                cv2.circle(img,(x,y),1,[0,255,0],-1)
                            else: # object out of the reange of the sensor; cannot say the type of the object.
                                objTYPE[fr][numo] = 'notype'

                            """
                            Consider for process all objects that are not 'notype'
                            """
                            if objTYPE[fr][numo] != 'notype':
                                                
                                # Get center of bounding box of tracked object.
                                bbox_x, bbox_y = centerBB(x_ul, y_ul, x_lr, y_lr)

                                if init.use_rcnn:
                                    nametext = names[numo]
                                else:
                                    nametext = objtype[:-1] + str(numo)

                                # Visualize the type of the object on the plotted image.
                                cv2.putText(img, nametext, (int(bbox_x), int(bbox_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,30), 2, cv2.LINE_AA)

                                # Keep the depth of the central point of each objects, w/o any noise.
                                '''  
                                depth_object[numo] = dimg[int(bbox_y)][int(bbox_x)]
                                flag_depth = True
                                dx, dy = abs(x_lr-x_ul), abs(y_lr-y_ul)
                                cx, cy = int(bbox_x), int(bbox_y)
                                dim = 3
                                i = (dim-1)/2
                                while ((flag_depth) and (dim < min(dx,dy)) and (i < dim)):
                                    if dimg[int(cy-(dim-1)/2)][int(cx-i)] != 0:
                                        depth_object[numo] = dimg[int(cy-(dim-1)/2)][int(cx-i)]
                                        break
                                    if dimg[int(cy+(dim-1)/2)][int(cx-i)] != 0:
                                        depth_object[numo] = dimg[int(cy+(dim-1)/2)][int(cx-i)]
                                        break
                                    if dimg[int(cy-i)][int(cx-(dim-1)/2)] != 0:
                                        depth_object[numo] = dimg[int(cy-i)][int(cx-(dim-1)/2)]
                                        break
                                    if dimg[int(cy-i)][int(cx+(dim-1)/2)] != 0:
                                        depth_object[numo] = dimg[int(cy-i)][int(cx+(dim-1)/2)]
                                        break
                                    if (i < dim):
                                        i += 1
                                    else:
                                        dim += 2
                                        i = (dim-1)/2
                                '''
                                    
                                xs, ys = sizeBB(x_ul, y_ul, x_lr, y_lr)

                                PrevObj[fr][Rnumo][0], PrevObj[fr][Rnumo][1], PrevObj[fr][Rnumo][2], PrevObj[fr][Rnumo][3] = \
                                    bbox_x, bbox_y, xs, ys
                                #ObjType[fr][numo] = nametext
                                
                                if objlist_pervideo[numo] == []:
                                    objlist_pervideo[numo] = nametext# + str(numo) # Confusion with the printed name and the saved name (including once more the obj ID)
                                    #print(objlist_pervideo)

                                # To find the exact position of the Lqsr and the objects to be used by their ID in every video (in reference to the total amount of objects)
                                # Pointer array for only the objects we are interested in.
                                ObjID[fr][Rnumo] = this_object_id
                                Rnumo += 1


                    # ************** List of QSR ************** 
                    # ['argd', 'argprobd', 'cardir', 'mos', 'mwe', 'qtcbcs', 'qtcbs',
                    # 'qtccs', 'ra', 'rcc2', 'rcc3', 'rcc4', 'rcc5', 'rcc8', 'tpcc']
                    qsr = 'rcc5'
                    
                    if Rnumo >= 2:
                        
                        # Find all pairs of interacting objects in the scene
                        #for pair in itertools.combinations(range(numobjects), 2):
                        for pair in itertools.combinations(range(Rnumo), 2):
                            # Find the ID
                            ID0 = ObjID[fr][pair[0]]-1
                            ID1 = ObjID[fr][pair[1]]-1

                            objName0 = objlist_pervideo[ID0]
                            objName1 = objlist_pervideo[ID1]

                            #Lo0 = copy.deepcopy(Lqsr[ID0])
                            #Lo1 = copy.deepcopy(Lqsr[ID1])
                            #Lo0 = Lqsr[ID0]
                            #Lo1 = Lqsr[ID1]

                            

                            try:
                                interaction = objlist_interactions.index([objName0, objName1])
                                Lo0 = copy.deepcopy(Linteractions[interaction][0])
                                Lo1 = copy.deepcopy(Linteractions[interaction][1])
                                objName0 = objlist_interactions[interaction][0]
                                objName1 = objlist_interactions[interaction][1]
                                flagA = True
                            except ValueError:
                                try:
                                    #interaction = dict_combinations[str(ID1)+','+str(ID0)]
                                    interaction = objlist_interactions.index([objName1, objName0])
                                    Lo0 = copy.deepcopy(Linteractions[interaction][1])
                                    Lo1 = copy.deepcopy(Linteractions[interaction][0])
                                    objName0 = objlist_interactions[interaction][1]
                                    objName1 = objlist_interactions[interaction][0]
                                    flagA = False
                                except ValueError:
                                    interaction = objlist_interactions.index([])
                                    objlist_interactions[interaction] = [objName0, objName1]
                                    Lo0 = copy.deepcopy(Linteractions[interaction][0])
                                    Lo1 = copy.deepcopy(Linteractions[interaction][1])
                                    objName0 = objlist_interactions[interaction][0]
                                    objName1 = objlist_interactions[interaction][1]
                                    flagA = True


                            Lo0, Lo1 = AddQSR(fr, Lo0, objName0, PrevObj[fr][pair[0]], Lo1, objName1, PrevObj[fr][pair[1]])

                            Lo0, Lo1, img = ConvexConcaveInteraction(deletelonely, detect3D, ID0, ID1, Lo0, Lo1, dict_combinations, objTYPE[fr], objTHRES, qsr, fr,\
                                PrevObj, objName0, objName1, pair[0], pair[1], img)


                            #Lqsr[ID0] = copy.deepcopy(Lo0)
                            #Lqsr[ID1] = copy.deepcopy(Lo1)

                            if flagA:
                                Linteractions[interaction][0] = copy.deepcopy(Lo0)
                                Linteractions[interaction][1] = copy.deepcopy(Lo1)
                            else:
                                Linteractions[interaction][1] = copy.deepcopy(Lo0)
                                Linteractions[interaction][0] = copy.deepcopy(Lo1)

                            # Find interactivity
                            #count_spatial_nodes = AddObjInteractivity(count_spatial_nodes, qsr, Lo0, Lo1, objName0, objName1, flag_interactivity = True)
                            #obj_relations_prev = count_spatial_nodes

                            #print_obj_list, include_more = PrintInteractivities_2obj(task, objName0, objName1, Rnumo, print_obj_list, \
                            #    enable_obj_interactivity = print_objs_inter, save_affordances = save_affordances)
                    else:
                        ID0 = ObjID[fr][0]-1
                        objName0 = objlist_pervideo[ID0]
                        obj2inc = '|' + str(objName0) +'|'
                        if obj2inc not in print_obj_list:
                            print_obj_list.append(obj2inc)
                        count_spatial_nodes = obj_relations_prev
                    print_objs_inter = False

                    '''
                    for obs in range(Rnumo):
                        hand_relations = 0
                        IDobjs = ObjID[fr][obs]-1

                        objName = objlist_pervideo[IDobjs]
                        handName0 = 'hand0'
                        handName1 = 'hand1'

                        #Lo = copy.deepcopy(Lqsr[IDobjs])
                        #Loh0 = copy.deepcopy(Lqsr_hands[0])
                        #Loh1 = copy.deepcopy(Lqsr_hands[1])
                        Lobjh0 = Lqsr_hands[IDobjs][0][0]
                        Lobjh1 = Lqsr_hands[IDobjs][1][0]
                        Loh0 = Lqsr_hands[IDobjs][0][1]
                        Loh1 = Lqsr_hands[IDobjs][1][1]

                        
                        Lobjh0, Loh0 = AddQSR(fr, Lobjh0, objName, PrevObj[fr][obs], Loh0, handName0, PrevHands[fr][0])
                        #print('h0', Lo0[len(Lo0)-1])
                        Lqsr_hands[IDobjs][0][0] = Lobjh0
                        Lqsr_hands[IDobjs][0][1] = Loh0
                        #hand_relations, count_spatial_nodes = AddHandInteractivity(hand_relations, count_spatial_nodes, qsr, Lobjh0, Loh0, objName, handName0, flag_interactivity = find_interactivity)
                        
                        # Do the same for the second hand
                        Lobjh1, Loh1 = AddQSR(fr, Lobjh1, objName, PrevObj[fr][obs], Loh1, handName1, PrevHands[fr][1])
                        #print('h1', Lo0[len(Lo0)-1])
                        Lqsr_hands[IDobjs][1][0] = Lobjh1
                        Lqsr_hands[IDobjs][1][1] = Loh1
                        #hand_relations, count_spatial_nodes = AddHandInteractivity(hand_relations, count_spatial_nodes, qsr, Lobjh1, Loh1, objName, handName1, flag_interactivity = find_interactivity)
                   
                        #print_int_list = PrintInteractivities_hands(task, objName, handName0, handName1, print_int_list, include_more, enable_hand_interactivity = print_interactions, save_affordances = save_affordances)


                        # Change the color of the bounding boxes if a RCC itneraction occures 
                        qsr_value = WhichInteraction(qsr, fr,  PrevObj[fr][obs], PrevHands[fr][0], objName, handName0)
                        if qsr_value !='dc':
                            x0_center, y0_center, x0_size, y0_size = \
                                PrevObj[fr][obs][0], PrevObj[fr][obs][1], PrevObj[fr][obs][2], PrevObj[fr][obs][3]
                            cv2.line(img, (int(x0_center+x0_size/2),int(y0_center+y0_size/2)),(int(x0_center-x0_size/2),int(y0_center+y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center+x0_size/2),int(y0_center+y0_size/2)),(int(x0_center+x0_size/2),int(y0_center-y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center-x0_size/2),int(y0_center-y0_size/2)),(int(x0_center-x0_size/2),int(y0_center+y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center-x0_size/2),int(y0_center-y0_size/2)),(int(x0_center+x0_size/2),int(y0_center-y0_size/2)), (0,0,255), 4)
                            x1_center, y1_center, x1_size, y1_size = \
                                PrevHands[fr][0][0], PrevHands[fr][0][1], PrevHands[fr][0][2], PrevHands[fr][0][3]
                            cv2.line(img, (int(x1_center+x1_size),int(y1_center+y1_size)),(int(x1_center-x1_size),int(y1_center+y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center+x1_size),int(y1_center+y1_size)),(int(x1_center+x1_size),int(y1_center-y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center-x1_size),int(y1_center-y1_size)),(int(x1_center-x1_size),int(y1_center+y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center-x1_size),int(y1_center-y1_size)),(int(x1_center+x1_size),int(y1_center-y1_size)), (0,0,255), 4)

                        qsr_value = WhichInteraction(qsr, fr, PrevObj[fr][obs], PrevHands[fr][1], objName, handName1)
                        if qsr_value !='dc':
                            x0_center, y0_center, x0_size, y0_size = \
                                PrevObj[fr][obs][0], PrevObj[fr][obs][1], PrevObj[fr][obs][2], PrevObj[fr][obs][3]
                            cv2.line(img, (int(x0_center+x0_size/2),int(y0_center+y0_size/2)),(int(x0_center-x0_size/2),int(y0_center+y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center+x0_size/2),int(y0_center+y0_size/2)),(int(x0_center+x0_size/2),int(y0_center-y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center-x0_size/2),int(y0_center-y0_size/2)),(int(x0_center-x0_size/2),int(y0_center+y0_size/2)), (0,0,255), 4)
                            cv2.line(img, (int(x0_center-x0_size/2),int(y0_center-y0_size/2)),(int(x0_center+x0_size/2),int(y0_center-y0_size/2)), (0,0,255), 4)
                            x1_center, y1_center, x1_size, y1_size = \
                                PrevHands[fr][1][0], PrevHands[fr][1][1], PrevHands[fr][1][2], PrevHands[fr][1][3]
                            cv2.line(img, (int(x1_center+x1_size),int(y1_center+y1_size)),(int(x1_center-x1_size),int(y1_center+y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center+x1_size),int(y1_center+y1_size)),(int(x1_center+x1_size),int(y1_center-y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center-x1_size),int(y1_center-y1_size)),(int(x1_center-x1_size),int(y1_center+y1_size)), (0,0,255), 4)
                            cv2.line(img, (int(x1_center-x1_size),int(y1_center-y1_size)),(int(x1_center+x1_size),int(y1_center-y1_size)), (0,0,255), 4)
                        

                        if Rnumo >= 2:
                            if obs == 0:
                                rel_handobj0 = hand_relations
                                obj0_id = IDobjs
                            else:
                                rel_handobj1 = hand_relations
                                obj1_id = IDobjs
                        else:
                            if IDobjs == obj0_id:
                                count_spatial_nodes += rel_handobj1
                            else:
                                count_spatial_nodes += rel_handobj0

                    
                    print_interactions = False
                    include_more = False

                    '''

                    
                    if find_interactivity:
                        if plot_flag:
                            # Create plots of interactivity
                            frame_plot.append(fr)
                            spatial_plot.append(count_spatial_nodes)
                            # Plotting the growth of the activity graph
                            fig.clear()
                            plt.plot(frame_plot, spatial_plot)
                            plt.ylim((0,70))
                            plt.xlim((0,500))
                            plt.ylabel('Spatio-temporal interactions')
                            plt.xlabel('Frames')
                            plt.pause(0.0005)
                            plt.savefig('/tmp/Plot'+str(fr)+'.png')

                            cv2.imwrite('/tmp/Img'+str(fr)+'.png', img)      

                    # ------------------- End of Objects -------------------#

                    if len(Jbody)!=0:
                        cv2.line(img, (lsx,lsy),(rhpx,lsy), (0,255,0), 4)
                        cv2.line(img, (lsx,lsy),(lsx,rhpy), (0,255,0), 4)
                        cv2.line(img, (rhpx,rhpy),(rhpx,lsy), (0,255,0), 4)
                        cv2.line(img, (rhpx,rhpy),(lsx,rhpy), (0,255,0), 4)
        
                    # ------------------- Create Graph ------------------- # 
                    G_hands = BuildGraph(Jhands, 'HND') 
                    G_head = BuildGraph(Jhead, 'HD')
                    G_body = BuildGraph(Jbody, 'B')
                    G_feet = BuildGraph(Jfeet, 'F')
                    G_head_hands = BuildGraph(J_head_hands, 'H&H')  
                    G_hands_feet = BuildGraph(J_hands_feet, 'H&F')

                    # ---- Plot Graph ---- #     
                    if init.show_graph:
                        nx.draw(G, with_labels=True)
                        plt.show()
                    # ----------------- End Graph ----------------- #

                    if show_image:
                        cv2.imshow('CAD-120: QSR for bounding boxes',img) 
                        #cv2.imshow('cropped object', thres11)
                        #cv2.imshow('cropped gray', crop_gray11)
                        #cv2.imshow('contours', drawing)
                        #cv2.imshow('median filtering', med_dimg)
                        #cv2.imshow('Depth', dimg)
                    #cv2.imshow('Depth COLOR', rgbdimg)

                    #plt.pause(0.0005)

                    #if cv2.waitKey(33) == ord('a'):
                    #    Visualize_plot = True
                    #elif cv2.waitKey(30) & 0xff == 27:
                    #    break

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

                # For every video capture the whole graph
                now = datetime.datetime.now()
                st1 = now.strftime("%Y_%m_%d_%H_%M_%S")
                pf = "qsr_figs/"
                if init.save_video_graph:
                    if not os.path.exists(pf):
                        os.makedirs(pf)

                pathsave = pf + "vhands_" + st1 + ".pdf"
                #ShowAndSaveGraph(pathsave, J_hands, G_hands)
                pathsave = pf + "vhandsfeet_" + st1 + ".pdf"
                #ShowAndSaveGraph(pathsave, J_hands_feet, G_hands_feet)
                ShowGraph(G_head)
                ShowGraph(G_body)
                ShowGraph(G_feet)
                ShowGraph(G_head_hands)

                print('All Done')
                Visualize_plot = False
                cv2.destroyAllWindows()
                plt.close(fig)


                PATH_GRAPH = '/home/scat/OneDrive/Github/object_affordances/data/exp_data_depth/' + unique_graph_name + '/graphs/'
                if not os.path.exists(PATH_GRAPH):
                    os.mkdir(PATH_GRAPH)

                for l in range(len(Linteractions)):
                    # Excluding NULL graphs
                    if (Linteractions[l][0]!=[] and Linteractions[l][1]!=[]):
                        Lo0 = Linteractions[l][0]
                        Lo1 = Linteractions[l][1]
                        objName0 = objlist_interactions[l][0]
                        objName1 = objlist_interactions[l][1]

                        print(objName0, objName1)


                        graph = ParseQSR(qsr, Lo0, Lo1, objName0, objName1, show_qsr, median_filtering, filtering_window)
                        F, spatial_F, unique_spatial, spatial_Fobjs = GraphFeatures(graph, show_qsr, conceptual_PO, unique_graph_name, PATH_GRAPH)

                        # If there is some temporal information then the objects are being used, hence we consider their affordances.
                        if F != []:
                            F_list.append(F)
                            spatial_F_list.append(spatial_F)
                            unique_spatial_list.append(unique_spatial)
                            spatial_Fobjs_list.append(spatial_Fobjs)

                            video_name_list.append(task)
                            video_task_list.append(act)
                            video_objs_list.append([ objName0, objName1 ])

                '''
                for lists in itertools.combinations(range(len(Lqsr)), 2):
                    # Excluding NULL graphs
                    if (Lqsr[lists[0]]!=[] and Lqsr[lists[1]]!=[]):
                        Lo0 = Lqsr[lists[0]]
                        Lo1 = Lqsr[lists[1]]
                        objName0 = objlist_pervideo[lists[0]]
                        objName1 = objlist_pervideo[lists[1]]

                        graph = ParseQSR(qsr, Lo0, Lo1, objName0, objName1, show_qsr, median_filtering)
                        F, spatial_F, unique_spatial, spatial_Fobjs = GraphFeatures(graph, show_qsr, conceptual_PO, unique_graph_name)

                        print(lists[0], lists[1])
                        print(objName0, objName1)
                        print(spatial_F)
                        print(spatial_Fobjs)

                        # If there is some temporal information then the objects are being used, hence we consider their affordances.
                        if F != []:
                            F_list.append(F)
                            spatial_F_list.append(spatial_F)
                            unique_spatial_list.append(unique_spatial)
                            spatial_Fobjs_list.append(spatial_Fobjs)

                            video_name_list.append(task)
                            video_task_list.append(act)
                            video_objs_list.append([ objName0, objName1 ])
                '''
                
                # UNCOMMENT THIS to include the information of the hands
                '''
                objid = -1
                for lh_obj in Lqsr_hands:
                    objid += 1
                    for h in range(2):
                        if (lh_obj[h][0]!=[] and lh_obj[h][1]!=[]):
                            Lo = lh_obj[h][0]
                            Lh = lh_obj[h][1]
                            
                            
                            F, spatial_F, unique_spatial, spatial_Fobjs = ParseQSR(qsr, Lo, Lh, objlist_pervideo[objid], 'hand'+str(h), False)

                            F_list.append(F)
                            spatial_F_list.append(spatial_F)
                            unique_spatial_list.append(unique_spatial)
                            spatial_Fobjs_list.append(spatial_Fobjs)

                            video_name_list.append(task)
                            video_task_list.append(act)
                            video_objs_list.append([ objlist_pervideo[objid], 'hand'+str(h) ])
                '''

    # Find the ED between every pair of video.
    len_video = list(range(len(F_list)))
    video_comb = combinations(len_video, 2)
    for i in list(video_comb):
        v0, v1 = i[0], i[1]
        F = F_list[v0]
        F1 = F_list[v1]
        spatial_F = spatial_F_list[v0]
        spatial_F1 = spatial_F_list[v1]
        unique_spatial = unique_spatial_list[v0]
        unique_spatial1 = unique_spatial_list[v1]
        spatial_Fobjs = spatial_Fobjs_list[v0]
        spatial_Fobjs1 = spatial_Fobjs_list[v1]

        #print(F, F1, spatial_F, spatial_F1, unique_spatial, unique_spatial1, spatial_Fobjs, spatial_Fobjs1)
        if spatial_Fobjs != [] and spatial_Fobjs1 != []:
            ED = MyEDCalculator(F, F1, spatial_F, spatial_F1, unique_spatial, unique_spatial1, spatial_Fobjs, spatial_Fobjs1, 0)
            # Save in files:
            if init.save_clustering_data_files:
                WriteInFile(filename = init.ED_file, delimiter = ';', data = [ED])
                WriteInFile(filename = init.video_name_file, delimiter = ';', data = [video_name_list[v0], video_name_list[v1]])
                WriteInFile(filename = init.video_task_file, delimiter = ';', data = [video_task_list[v0], video_task_list[v1]])
                WriteInFile(filename = init.video_objs_file, delimiter = ';', data = [video_objs_list[v0], video_objs_list[v1]])


    # Save list in file of interactive joints and objects 
    if init.write_in_file:           
        f.write("Hands list: %s\r\n" % Jhands)
        f.write("Head list: %s\r\n" % Jhead)
        f.write("Body list: %s\r\n" % Jbody)
        f.write("Head and Hands list: %s\r\n" % J_head_hands)
        f.write("Hands and Feet list: %s\r\n" % J_hands_feet)

    # Save total graphs in qsr_figs/ directory
    now = datetime.datetime.now()
    st1 = now.strftime("%Y_%m_%d_%H_%M_%S")
    pf = "qsr_figs/"
    if init.save_video_graph:
        if not os.path.exists(pf):
            os.makedirs(pf)
    pathsave = pf + "hands_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, Jhands, G_hands)
    pathsave = pf + "head_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, Jhead, G_head)
    pathsave = pf + "body_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, Jbody, G_body)
    pathsave = pf + "feet_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, Jfeet, G_feet)
    pathsave = pf + "head_hands_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, J_head_hands, G_head_hands)
    pathsave = pf + "hands_feet_" + st1 + ".pdf"
    ShowAndSaveFinalGraph(pathsave, J_hands_feet, G_hands_feet)

    if init.write_in_file:    
        f.close()
    if save_affordances:
        fa.close() # close affordance file


