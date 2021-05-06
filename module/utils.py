#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.

import cv2
import os
import scipy
import skimage
import time
import re
import ast
import io

import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from tqdm import tqdm
from itertools import product, combinations
from scipy.ndimage import convolve
from skimage.measure import label, regionprops_table
from skimage.morphology import reconstruction
from numpy import matlib

def plotAPCA(op, X1, PC, Q, anomal_occur): 
    
    """
    Plot APCA result 
    
    Args : 
        X1
        PC
        Q
        
    Returns : 
        None 
    
    """
    
    fig = plt.figure(figsize=(12, 8))

    x0 = op['IND_x0']
    x1 = op['IND_x1']
    plt.plot(x0, Q['Qdist'][x0], 'bo', linewidth=2, markersize=8)
    plt.plot(x1, Q['Qdist'][x1], 'v', linewidth=2, markersize=8)

    alpha = 0.95
    SD_type = 'Z-score'
    PCA_par = {}
    PCA_par['n_stall'] = 3;

    IND_abnor = np.argwhere(PC['Updated']==0)
    IND_nor = np.argwhere(PC['Updated']==1)
    plt.plot(IND_abnor,Q['Qdist'][IND_abnor],'rs',linewidth=2, markersize=8) # Outlier and faulty samples
    plt.plot(IND_nor,Q['distcrit'][IND_nor][:,:, 0],'k-',linewidth=5, markersize=8) # Outlier and faulty samples
    plt.axvline(x = IND_nor[-1, 0], color = 'red', linestyle ='--', label='Structural Damage Detected')
    plt.axvline(x = anomal_occur, color = 'blue', linestyle =':', label='Structural Damage Occured')
    plt.ylabel('Q-statistics (SPE)', fontsize=20)
    plt.title('slab_no:' + '# Alpha='+str(alpha)+str(SD_type)+'# stall:'+str(PCA_par['n_stall']),fontsize=15,fontweight='bold')
    plt.legend(fontsize='x-large')
    plt.show
    
    

def plotData(op, disp_sns_num):
    
    """
    Plot temperature and displacement graph from APCA data 
    
    Args :
        op (dict) : APCA data 
        
    Returns : 
        None 
        
    """
    fig, ax1 = plt.subplots(figsize=(10, 10))
    color = 'tab:red'
    ax1.set_xlabel('time', fontsize = 20.0)
    ax1.set_ylabel('Temperaure(Celcius)', color=color, fontsize = 20.0)
    ax1.plot(op['t'].T, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color, labelsize = 24)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Displacement(mm)', color=color, fontsize = 20.0)  # we already handled the x-label with ax1
    ax2.plot(op['f'], linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color, labelsize = 24)
    ax2.legend(disp_sns_num, fontsize=15)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped



def cvtDataForAPCA(sns_dict, disp_sns_num, trainig_range, 
                   tmp_sns_num, anomal_occur = 0, anomal_disp = 0):
    
    
    """
    Args : 
        sns_dict (dictionary) : 
        disp_sns_num (string or list) : sensor num where displacement data to be imported 
        training_range (int or float) : num of data to be used for training. 
                                        Ininitial count starts from the first idx
        tmp_sns_num (string) : sensor num where temperature data to be imported 
        anomal_occur (int or float) : datetime when anomaly occurs 
                                      if not occur, it is set to zero. 
        anomal_disp (int or float) : displacement amount to be shifted from the original displacement 
                                    if it is set to None, the displacement will be fixed at anomal_occur
    Returns : 
        op (dictionary) : Python data dictionary prepared for APCA 
    
    """
    disp_data_list = []
    temp_data_list = []
    ite = 0
    for date, time in sns_dict[tmp_sns_num]['data'].items():
        for time_, dir_ in time.items():
            
            if ite > anomal_occur: 
                anomal_disp_ = anomal_disp 
            else : 
                anomal_disp_ = 0
            try:
                temp = dir_['temp']
            except :
                temp = None

            try:
                disp_list = []
                for idx, sns_num in enumerate(disp_sns_num) : 

                    if sns_dict[sns_num]['disp_dir'] == 'Forward' : 
                        disp_list.append(-(sns_dict[sns_num]['data'][date][time_]['x_disp'] + anomal_disp_))
                    else : 
                        disp_list.append(sns_dict[sns_num]['data'][date][time_]['x_disp'] + anomal_disp_)
            except :
                disp_list = None

            if temp and disp_list: 
                ite += 1
                temp_data_list.append(temp)  
                disp_data_list.append(disp_list) 
                
    op={}
    op['t'] = np.array(temp_data_list, dtype=np.float32).squeeze()[:, np.newaxis].T
    op['f'] = np.array(disp_data_list, dtype=np.float32).squeeze()
    op['IND_x0'] = np.arange(0, trainig_range)
    op['IND_x1'] = np.arange(trainig_range, len(disp_data_list))
    
    return op


### initialize a dictionary for analysis table 
def snsInfoToDict(sns_info): 
    
    """
    Args : 
        sns_info (pd.Dataframe) : 
        
    Returns : 
        sns_dict (dictionary) : dictionary contains sensor information 
    """
    sns_dict = {} 

    for idx, row in sns_info.iterrows():
        sns_id = str(row['Sensor ID']).zfill(3)
        sns_dict[sns_id] = {}
        sns_dict[sns_id]['type'] = row['Sensor Type']
        sns_dict[sns_id]['pier']= row['Pier ']
        sns_dict[sns_id]['slab']= row['Slab No']
        sns_dict[sns_id]['block']= row['Block']
        sns_dict[sns_id]['sns_dir']= row['Sensor Direction']
        sns_dict[sns_id]['sns_loc']= row['Sensor Location']
        sns_dict[sns_id]['disp_dir']= row['Displacement Direction']
        sns_dict[sns_id]['install_year']= row['Sensor Installation Year']
        sns_dict[sns_id]['data'] = {}
        
    return sns_dict

def getDirList(sns_info, dataset_folder, ana_periold, sns_type = None) : 
    
    """
    Args : 
        sns_info (pd.Dataframe) : sensor information in pandas dataframe
        dataset_folder (string) : dataset folder root directory
        ana_period (list) : Python list including start and end date of analysis
        sns_type (string) : data type, 'Img'  or 'Tmp'
    
    Returns : 
        data_list (list) : Python list including data director list under given condition 
    
    """
    
    sns_list = sns_info[(sns_info['Sensor Type'] == sns_type)]['Sensor ID']
    data_list = []

    for sensor_id in tqdm(sns_list,  desc="Searching for data under subfolders"):
        # check installation year 

        install_year = sns_info[(sns_info['Sensor ID'] == sensor_id)]['Sensor Installation Year'].values[0]
        foler_name = osp.join(dataset_folder, str(install_year), 'data')
        
        if sns_type == 'Img' :
            sns_data_list = glob(osp.join(foler_name, sns_type + '_' + str(sensor_id).zfill(3)+'*.jpg'))
        elif sns_type == 'Tmp' :
            sns_data_list = glob(osp.join(foler_name, sns_type + '_' + str(sensor_id).zfill(3)+'*.tpr'))

        for data_dir in sns_data_list: 
            data_date = int(osp.basename(data_dir)[8:16])
            if (data_date >= ana_periold[0]) and (data_date <= ana_periold[1]):
                data_list.append(data_dir)
                
    data_list.sort()
    

    return data_list

def dirListToDict(sns_dict, data_list):
    
    """
    Args : 
        sns_dict (dictionary) : dictionary contains sensor information 
        data_list (list) : Python list including data director list under given condition 
        
    Returns :
        sns_dict (dictionary) : updated dictionary with data directories 
    
    """
    
    for data_dir in data_list: 
        
        '''
        Example file name : Img_038_20200831_230100.jpg
        
            data_basename[4:7]  : 038 sns_id
            data_basename[8:16] : 20200831 date
            data_basename[17:23] :  230100 time 
            
        Since data acquisition is conducted once an hour, 
        the time will be trucated to only have hour information 
        
        '''
        
        data_basename = osp.basename(data_dir)
        sns_type = data_basename[:3]
        sns_id = data_basename[4:7]
        date = data_basename[8:16]
        time = data_basename[17:19] + '0000'
        
        # if date does not exist, create a dict 
        if date not in sns_dict[sns_id]['data'] : 
            sns_dict[sns_id]['data'][date] = {}
            
        # if time does not exist, create a dict 
        if time not in sns_dict[sns_id]['data'][date] : 
            sns_dict[sns_id]['data'][date][time] = {}
        
        # if time does not exist, create a dict 
        if sns_type not in sns_dict[sns_id]['data'][date][time] : 
            sns_dict[sns_id]['data'][date][time][sns_type] = {}
        
        sns_dict[sns_id]['data'][date][time][sns_type] = data_dir
        
    return sns_dict

def tmpListToDict(sns_dict, data_list):
    
    """
     Args : 
        sns_dict (dict) : 
        data_list (list) : 
    
    Returns : 
        sns_dict (dict) : 
         
    """
    
    for tmp_file in data_list:
        
        tmp_measured = None
        f = io.open(tmp_file, mode="r", encoding="utf-8")
        txt = f.read()
        if txt :
            tmp_measured = str2array(txt)[1]
            
        
        tmp_basename = osp.basename(tmp_file)
        sns_type = tmp_basename[:3]
        sns_id = tmp_basename[4:7]
        date = tmp_basename[8:16]
        time = tmp_basename[17:23]

        # if date does not exist, create a dict 
        if date not in sns_dict[sns_id]['data'] : 
            sns_dict[sns_id]['data'][date] = {}
            
        # if time does not exist, create a dict 
        if time not in sns_dict[sns_id]['data'][date] : 
            sns_dict[sns_id]['data'][date][time] = {}
        
        # if time does not exist, create a dict 
        if sns_type not in sns_dict[sns_id]['data'][date][time] : 
            sns_dict[sns_id]['data'][date][time][sns_type] = {}


        sns_dict[sns_id]['data'][date][time]['temp'] = tmp_measured
        
    return sns_dict

    
def saveCDResult(img_path, dataset_folder, param_config) : 
    
    """
     Args : 
        img_path (string) : 
        dataset_folder (string) : 
        param_config (dict) :
    
    Returns : 
        None (This function only saves circle detection results in jpg format.)
         
    """
    
    img = imread(img_path)
    img_basename = osp.basename(img_path)
    img_copy = img.copy()
    
    sns_id = img_basename[4:7]
    params = param_config[sns_id]
    
    min_rad = params.get("min_rad",70)
    max_rad = params.get("max_rad",100)
    sensitivity = params.get("sensitivity",0.98)
    gamma = float(params.get("gamma", 1.0))
    binarization = params.get("binarization", 0)
    
    result_folder = osp.join(dataset_folder, '2020', 'result', sns_id) # to do : make it not hardcoding
    Path(result_folder).mkdir(parents=True, exist_ok=True)


    if gamma != 1.0 :  
        img = adjust_gamma(img, gamma=gamma)

    if binarization : 
        img = adaptiveThreshold_3ch(img, min_rad)

    centers, r_estimated, metric = imfindcircles(img, 
                                                 [min_rad, max_rad],
                                                sensitivity = sensitivity)
    circles = np.concatenate((centers, r_estimated[:,np.newaxis]), axis = 0).T
    circles = np.squeeze(circles)
    if len(circles) > 4: 
        circles = find_valid_dest_circles(circles)

    if circles is not None: 
        # Convert the circle parameters a, b and r to integers. 
        circles = np.uint16(np.around(circles)) 
        if circles.ndim ==1 : 
            circles = circles[np.newaxis, :]
        for pt in circles: 
            a, b, r = pt[0], pt[1], pt[2]   
            # Draw the circumference of the circle. 
            cv2.circle(img_copy, (a, b), r, (0, 255, 0), 2) 
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img_copy, (a, b), 1, (0, 0, 255), 3) 

        cv2.imwrite(osp.join(result_folder, img_basename), img_copy)
    else:
        print("circles is not detected")
        
        
def cvtClcToDict(sns_dict, circles_list): 
    
    """
    Args :
        sns_dict :
        circles_list :
        
    Returns : 
        sns_dict : 
    
    """
    
    for clc in circles_list: 
        sns_num = clc[0]
        date = clc[1]
        time = clc[2][:2] + '0000'
        sns_dict[sns_num]['data'][date][time]['circles'] = date = clc[3]
        
    return sns_dict


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    #apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def adaptiveThreshold_3ch(img, kernel_size) : 

    img[:,:,0] = cv2.adaptiveThreshold(img[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,kernel_size,2)

    img[:,:,1] = cv2.adaptiveThreshold(img[:,:,1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,kernel_size,2)

    img[:,:,2] = cv2.adaptiveThreshold(img[:,:,2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,kernel_size,2)
    
    return img

## Function that find four valid circles in dest_circles
def find_valid_dest_circles(dest_circles):
    ## I. 상위 4개 원을 x value를 기준으로 sort 진행
    target = dest_circles[0:4] # 상위 4개 원 추출
    target = target[target[:,0].argsort()] # 이에 대해 sort

    x = target[:, 0] # extract x value of circles - get col vector
    y = target[:, 1] # extract y value of circles - get col vector

    ## II. 정렬 후 대각선으로 위치한 원들의 좌표를 더한 후...
    ## 각 좌표의 차이로 계산되어야해서 abs를 추가하였습니다. 
    x_dist = abs((x[0] + x[3]) - (x[1] + x[2]))
    y_dist = abs((y[0] + y[3]) - (y[1] + y[2]))
    

    # ISSUE: BH님이 말씀하신 criteria가 이것이 맞는지?
    # A : 넵 전달드린 criteria를 정확하게 작성해주셨습니다. 
    if (x_dist < 10.0) and (y_dist < 10.0): # nice case
        return dest_circles[0:4]
    
    ## III. 탐지된 원들 내에서 4개의 원을 추출하는 모든 경우의 수에 따라 원의 집합을 생성
    ## combination으로 계산되어도 문제 없습니다. 
    comb = combinations(dest_circles, 4) # [[[x1, y1, r1], [x2, y2, r2], ...], [4], [4], [4], ...]
    # IV. 각 원의 집합들에 대하여 2번 과정에 대한 연산을 수행
    dist_list = []
    elem_list = []
    dist_list_2 = []
    for elem in comb:
        dist_list_tmp = []
        # elem 이 제 컴퓨터에서는 tuple로 반환되서 40번째 줄에서 인덱싱이 안되더군요 
        # 그래서 np.array로 반환하였습니다. 
        elem = np.asarray(elem) 
        elem = elem[elem[:,0].argsort()] # 이에 대해 sort
        # ISSUE: 각 elem에 대해서 x_val 기준 정렬을 진행해야하는가?
        x = elem[:, 0] # extract x value of circles - get col vector
        y = elem[:, 1] # extract y value of circles - get col vector
        
        x_dist = abs((x[0] + x[3]) - (x[1] + x[2]))
        y_dist = abs((y[0] + y[3]) - (y[1] + y[2]))
        
        dist_list.append(x_dist + y_dist)
        
        dist_list_tmp.append(np.linalg.norm(target[0, :2] - target[1, :2]))
        dist_list_tmp.append(np.linalg.norm(target[1, :2] - target[2, :2]))
        dist_list_tmp.append(np.linalg.norm(target[2, :2] - target[3, :2]))
        dist_list_tmp.append(np.linalg.norm(target[3, :2] - target[0, :2]))
        dist_list_tmp = np.asarray(dist_list_tmp)
        elem_list.append(elem)
        dist_list_2.append(np.mean(dist_list_tmp)-112)
    
    # V. 4번 연산을 수행하여 가장 낮은 값을 갖는 집합을 반환 
    min_idx = np.argmin(dist_list_2)
    dest_circles_with_min_dist = elem_list[min_idx]
    
    return dest_circles_with_min_dist


def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    s=re.sub('\x00', '', s)
    return np.array(ast.literal_eval(s))



def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None
    

def normalizeControlPoints(pts):
    
    """
    This code is a Python implementation of Matlab code
    
    'images.geotrans.internal.normalizeControlPoints'
    
    Reference : 
    
    [1] Matlab Code Download link : https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46441/versions/3/previews/MarsRover2014/matlab/calibration/myNormalizeControlPoints.m/index.html
    
    [2] Hartley, R; Zisserman, A. "Multiple View Geometry in Computer Vision." Cambridge University Press, 2003. pg. 180-181.

    """
    
    # Define N, the number of control points
    N = pts.shape[0]

    # Compute [xCentroid,yCentroid]
    cent = np.mean(pts, 0)

    # Shift centroid of the input points to the origin.
    ptsNorm = np.zeros((4, 2))
    ptsNorm[:, 0] = pts[:, 0] - cent[0]
    ptsNorm[:, 1] = pts[:, 1] - cent[1]

    sumOfPointDistancesFromOriginSquared = np.sum(np.power(np.hypot(ptsNorm[:, 0], ptsNorm[:, 1]), 2))

    if sumOfPointDistancesFromOriginSquared > 0:
        scaleFactor = np.sqrt(2 * N) / np.sqrt(sumOfPointDistancesFromOriginSquared)
        
    #     else:
    # % If all input control points are at the same location, the denominator
    # % of the scale factor goes to 0. Don't rescale in this case.
    #         if isa(pts,'single'):
    #             scaleFactor = single(1);
    #         else:
    #             scaleFactor = 1.0;
    # % Scale control points by a common scalar scale factor
    
    ptsNorm = np.multiply(ptsNorm, scaleFactor)

    normMatrixInv = np.array([
        [1 / scaleFactor, 0, 0],
        [0, 1 / scaleFactor, 0],
        [cent[0], cent[1], 1]])

    return ptsNorm, normMatrixInv


def findProjectiveTransform(uv, xy):

    """
    This code is a Python implementation of Matlab code
    
    'findProjectiveTransform' in 'fitgeotrans' 
    
    Reference : to-be added here
    
    """
        
    uv, normMatrix1 = normalizeControlPoints(uv);
    xy, normMatrix2 = normalizeControlPoints(xy);

    minRequiredNonCollinearPairs = 4;
    M = xy.shape[0]
    x = xy[:, 0][:, np.newaxis]
    y = xy[:, 1][:, np.newaxis]
    vec_1 = np.ones((M, 1))
    vec_0 = np.zeros((M, 1))
    u = uv[:, 0][:, np.newaxis]
    v = uv[:, 1][:, np.newaxis]

    U = np.vstack([u, v])
    X = np.hstack([[x, y, vec_1, vec_0, vec_0, vec_0, np.multiply(-u, x), np.multiply(-u, y)],
                   [vec_0, vec_0, vec_0, x, y, vec_1, np.multiply(-v, x), np.multiply(-v, y)]]).squeeze().T

    # We know that X * Tvec = U
    if np.linalg.matrix_rank(X) >= 2 * minRequiredNonCollinearPairs:
        Tvec = np.linalg.lstsq(X, U, rcond=-1)[0]
    #     else :
    #         error(message('images:geotrans:requiredNonCollinearPoints', minRequiredNonCollinearPairs, 'projective'))

    #      We assumed I = 1;
    Tvec = np.append(Tvec, 1)
    Tinv = np.reshape(Tvec, (3, 3));

    Tinv = np.linalg.lstsq(normMatrix2, np.matmul(Tinv, normMatrix1), rcond=-1)[0]
    T = np.linalg.inv(Tinv)
    T = T / T[2, 2]

    return T

def getGrayImage(img) : 
   

    N = img.ndim
    if N == 3 : # If RGB Image, 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint8:    
            img = np.asarray(img, dtype = np.float)/255
    elif N == 2: 
        if img.dtype == np.uint8:    
            img = np.asarray(img, dtype = np.float)/255
    return img 


def imgradient(img): 
    # 하드 코딩 : Matlab Sobel Filter
    hy = -np.asarray([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])
    hx = hy.T
    
    Gx = convolve(img, hx, mode='nearest')
    Gy = convolve(img, hy, mode='nearest')
    
    gradientImg = np.hypot(Gx, Gy);
        
    return Gx, Gy, gradientImg


def getEdgePixels(gradientImg, edgeThresh): 
    Gmax = np.max(gradientImg[:])
    
    if not edgeThresh :
        # Python cv2 doen't take float image for otsu, 
        # Thus, we convert it into np.uint8 for thresholding for a while 
        edgeThresh, _ = cv2.threshold(np.asarray(gradientImg/Gmax*255, dtype=np.uint8),0,255, cv2.THRESH_OTSU)
        edgeThresh = edgeThresh/255

    t = Gmax * edgeThresh
    Ey, Ex = np.nonzero(gradientImg > t)

    return Ex, Ey


def bsxfun_1d(row, col, mode = 'times') : 
    
    mat_from_row = matlib.repmat(row, col.shape[0], 1)
    mat_from_col = matlib.repmat(col, row.shape[0], 1).T
    
    if mode == 'times':
        mat = np.multiply(mat_from_row, mat_from_col)
        
    elif mode == 'plus':
        mat = mat_from_row + mat_from_col
    
    return mat


def accum(accmap, a, size=None ): 
    
    if not size : 
        result = np.zeros((np.max(accmap[:, 0]), np.max(accmap[:, 1])), dtype=np.complex_) 
    
    else :
        result = np.zeros(size, dtype=np.complex_)

    # a = a[:, np.newaxis]
    np.add.at(result, (accmap[:, 1]-1, accmap[:, 0]-1), a)
    # concat = np.hstack((accmap, a))
    # concat = concat[concat[:, 0].argsort()] # 이에 대해 sort
    # df = pd.DataFrame(concat)

    # x = df.groupby([0, 1]).sum().reset_index()[range(3)] #range(5) adjusts ordering

    # x_0 = np.asarray(x[0], dtype = np.int64) - 1
    # x_1 = np.asarray(x[1], dtype = np.int64) - 1

    # result[x_1, x_0] = x[2]
    return result


def chaccum(A, radiusRange, method = 'phasecode',  objPolarity = 'bright', edgeThresh = [] ) : 
    
    maxNumElemNHoodMat = 1e6

    # all(A(:) == A(1));
    # Assume that image is not flat 

    # Get the input image in the correct format
    A = getGrayImage(A);

    ## Calculate gradient
    Gx, Gy, gradientImg = imgradient(A);

    Ex, Ey = getEdgePixels(gradientImg, edgeThresh);
    E = np.stack((Ey, Ex))

    # Not sure np.ravel_multi_index is equivalent with Matlab Function sub2ind
    idxE = np.ravel_multi_index(E, tuple(gradientImg.shape), mode='raise', order='C') 
    
    # Adding 0.0001 is to include end point of the radius
    radiusRange = np.arange(radiusRange[0], radiusRange[1]+0.0001, 0.5)

    if objPolarity == 'bright' :
        RR = radiusRange
    elif objPolarity == 'dark' :
        RR = - radiusRange

    ### This implementation only includes 'phasecode' mode 

    lnR = np.log(radiusRange);
    phi = ((lnR - lnR[0])/(lnR[-1] - lnR[0])*2*np.pi) - np.pi; # Modified form of Log-coding from Eqn. 8 in [3]

    Opca = np.exp(np.sqrt(-1+0j)*phi);
    w0 = Opca/(2*np.pi*radiusRange);

    xcStep = int(maxNumElemNHoodMat//RR.shape[0])

    lenE = Ex.shape[0]
    M, N = A.shape[0], A.shape[1]
    accumMatrix = np.zeros((M,N))
    
    for i in range(0, lenE+1, xcStep) : 
        
        
        Ex_chunk = Ex[i:int(np.min((i+xcStep-1,lenE)))]
        Ey_chunk = Ey[i:int(np.min((i+xcStep-1,lenE)))]
        idxE_chunk = idxE[i:int(np.min((i+xcStep-1,lenE)))]
        

        chunckX = np.divide(Gx[np.unravel_index(idxE_chunk, Gx.shape, 'C')],
                  gradientImg[np.unravel_index(idxE_chunk, gradientImg.shape, 'C')])

        chunckY = np.divide(Gy[np.unravel_index(idxE_chunk, Gy.shape, 'C')],
                  gradientImg[np.unravel_index(idxE_chunk, gradientImg.shape, 'C')])
        
        # Eqns. 10.3 & 10.4 from Machine Vision by E. R. Davies
        xc = matlib.repmat(Ex_chunk, RR.shape[0], 1).T + bsxfun_1d(-RR, chunckX) 
        yc = matlib.repmat(Ey_chunk, RR.shape[0], 1).T + bsxfun_1d(-RR, chunckY)

        xc = np.round(xc)
        yc = np.round(yc)

        w = matlib.repmat(w0, xc.shape[0], 1)

        ## Determine which edge pixel votes are within the image domain
        # Record which candidate center positions are inside the image rectangle.
        M, N = A.shape[0], A.shape[1]

        inside = (xc >= 1) & (xc <= N) & (yc >= 1) & (yc < M)
        # Keep rows that have at least one candidate position inside the domain.
        rows_to_keep = np.any(inside, 1);
        xc = xc[rows_to_keep,:];
        yc = yc[rows_to_keep,:];
        w = w[rows_to_keep,:];
        inside = inside[rows_to_keep,:];

        ## Accumulate the votes in the parameter plane
        xc = xc[inside]
        yc = yc[inside]

        xyc = np.asarray(np.stack((xc[:], yc[:])), dtype= np.int64)

        accumMatrix = accumMatrix + accum(xyc.T, w[inside], size=[M, N])

        del xc, yc, w # These are cleared to create memory space for the next loop. Otherwise out-of-memory at xc = bsxfun... in the next loop.
    
    return accumMatrix, gradientImg



def chcenters(accumMatrix, accumThresh) : 
    suppThreshold = accumThresh
    medFiltSize = 5
    kernel = np.ones((medFiltSize, medFiltSize),np.float32)/medFiltSize**2
    Hd = cv2.filter2D(accumMatrix,-1,kernel)

    suppThreshold = np.max((suppThreshold - np.spacing(suppThreshold), 0))   

    Hd = reconstruction(Hd-suppThreshold, Hd, method ='dilation')

    lm = scipy.ndimage.filters.maximum_filter(Hd, size=3)
    bw = lm > np.max(lm)/3
    label_bw = label(bw)

    s = regionprops_table(label_bw, properties=['label','centroid'])
    centers = np.stack((s['centroid-0' ], s['centroid-1']))
    
    centers_int = np.asarray(centers, dtype = np.int64)
    metric = Hd[centers_int[0], centers_int[1]]
    
    # Sort the centers in descending order of metric    
    
    metric_sorted=np.sort(metric)[::-1]
    sortIndx=np.argsort(metric) 
    centers_sorted = np.fliplr(np.stack((np.take(centers[0,:], sortIndx), np.take(centers[1,:], sortIndx))))
    
    return np.flipud(centers_sorted), metric_sorted


def chradiiphcode(centers, accumMatrix, radiusRange) :
    centers_int = np.asarray(centers, dtype = np.int64)
    cenPhase = np.angle(accumMatrix[centers_int[1], centers_int[0]])
    lnR = np.log(radiusRange);
    
    # Inverse of modified form of Log-coding from Eqn. 8 in [1]
    r_estimated = np.exp(((cenPhase + np.pi)/(2*np.pi)*(lnR[-1] - lnR[0])) + lnR[0]) 
    
    return r_estimated


def imfindcircles(A, radiusRange,  ObjectPolarity = 'bright', sensitivity = 0.95):

    [accumMatrix, gradientImg] = chaccum(A, radiusRange, objPolarity=ObjectPolarity);

    ## Estimate the centers
    accumThresh = 1 - sensitivity;

    accumMatrix = abs(accumMatrix);

    centers, metric = chcenters(accumMatrix, accumThresh)

    ## Assume Image is flat 
    ## Currently, Sigma is not specified 
    idx2Keep = np.nonzero(metric >= accumThresh)
    centers = centers[:, idx2Keep]
    metric = metric[idx2Keep]

    r_estimated = chradiiphcode(centers, accumMatrix, radiusRange)
    
    return centers, r_estimated, metric

def circle_detection_multi_thread(img_name, param_config):
    
    img_basename = osp.basename(img_name)
    sensor_num = str(img_basename[4:7])
    date = img_basename[8:8+8]
    time = img_basename[17:17+6]
    params = param_config[sensor_num]
    img = imread(img_name)
    sensitivity = params['sensitivity']
    circles = []
    try : 
        while len(circles) < 4 :
            centers, r_estimated, metric = imfindcircles(img, 
                                                         [params['min_rad'], params['max_rad']],
                                                         sensitivity = sensitivity)
            circles = np.concatenate((centers, r_estimated[:,np.newaxis]), axis = 0).T
            circles = np.squeeze(circles)
            sensitivity += 0.01
            if ((sensitivity > 1) and (len(circles) < 4)) or (len(circles) > 10): 
    #             width = img.shape[0]
    #             height = img.shape[1]
    #             radius = (params['max_rad']+params['min_rad'])/2
    #             step = radius*1.25
                circles = 'No circle is detected or Too manys are detected.'
                return sensor_num, date, time, circles
    except : 
        circles = 'circle detection is failed .'
        return sensor_num, date, time, circles
    
    circles = find_valid_dest_circles(circles).tolist()

    return sensor_num, date, time, circles