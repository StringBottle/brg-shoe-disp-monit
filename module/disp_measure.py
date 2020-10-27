#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.

import numpy as np
import cv2
from itertools import combinations 
from .utils import findProjectiveTransform, imfindcircles, find_valid_dest_circles, adjust_gamma, adaptiveThreshold_3ch

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (lm1, lm2) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    (rm1, rm2) = rightMost[np.argsort(rightMost[:, 1]), :]
    return np.array([lm1, lm2, rm1, rm2])

def homography_transformation(src_circles, dest_circles, p_length = 50) : 
    
    src_matrix = np.array([
        [src_circles[0][0], src_circles[0][1]],
        [src_circles[1][0], src_circles[1][1]],
        [src_circles[2][0], src_circles[2][1]],
        [src_circles[3][0], src_circles[3][1]]
    ])

    # x value를 기준으로 sort 진행
    src_matrix = order_points(src_matrix)
    dest_circles = order_points(dest_circles)


    # p_length에 따른 weight matrix 생성
    weight = np.array([
        [0, 0],
        [0, p_length/2],
        [p_length/2, 0],
        [p_length/2, p_length/2]
    ])
    
    # cv2.findHomography(src, desc, ...) 이용해서 호모그래피 matrix 찾기
    h_matrix = findProjectiveTransform(src_matrix, weight).T

    #############################################
    ## dest 전처리
    dest_vec1 = np.array([[dest_circles[0][0]], [dest_circles[0][1]], [1]])
    dest_vec2 = np.array([[dest_circles[1][0]], [dest_circles[1][1]], [1]])
    dest_vec3 = np.array([[dest_circles[2][0]], [dest_circles[2][1]], [1]])
    dest_vec4 = np.array([[dest_circles[3][0]], [dest_circles[3][1]], [1]])

    ## dest에 h_matrix 적용 (with inner product)
    dist_1 = np.dot(h_matrix, dest_vec1)
    dist_1 = dist_1 / dist_1[2]
    
    dist_2 = np.dot(h_matrix, dest_vec2)
    dist_2 = dist_2 / dist_2[2]

    dist_3 = np.dot(h_matrix, dest_vec3)
    dist_3 = dist_3 / dist_3[2]

    dist_4 = np.dot(h_matrix, dest_vec4)
    dist_4 = dist_4 / dist_4[2]

    ## 결과 도출 - result
    result = (dist_1 + dist_2 + dist_3 + dist_4) / 4 - np.array([[p_length/4], [p_length/4], [1]])

    return result[:2]


## Function with Path
def convert_by_path(dest_path, src_path):
    

    
    dest_img = cv2.imread(dest_path)
    dest_img = cv2.medianBlur(dest_img, 5)

    src_img = cv2.imread(src_path)
    src_img = cv2.medianBlur(src_img, 5)

    convert_by_img(dest_img, src_img)

## Function with ndarray(img)
def displacement_measure(dest_img,
                         src_img,   
                         params = None,
                         src_circles = None,
                        ):
    
   
    min_rad = params.get("min_rad",70)
    max_rad = params.get("max_rad",100)
    sensitivity = params.get("sensitivity",0.98)
    gamma = float(params.get("gamma", 1.0))
    binarization = params.get("binarization", 0)

    
    ## Parameters
    # constant for weight matrix
    # set max, min radius of finding circles
    
    if gamma != 1.0 :  
        dest_img = adjust_gamma(dest_img, gamma=gamma)
        src_img = adjust_gamma(src_img, gamma=gamma)
    
    if binarization : 
        dest_img = adaptiveThreshold_3ch(dest_img, min_rad)
        src_img = adaptiveThreshold_3ch(src_img, min_rad)
        
        
        
    
    if src_circles is None : 
#         grey_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        centers, r_estimated, metric = imfindcircles(src_img, 
                                                     [min_rad, max_rad],
                                                    sensitivity = sensitivity)

        src_circles = np.concatenate((centers, r_estimated[:,np.newaxis]), axis = 0).T
        src_circles = np.squeeze(src_circles)

    iter_count = 1
    num_centers = 0
    x_dist = 11
    y_dist = 11
    num_allowable_centers = 4
    dist_mean = 150

    while (x_dist > 5) or (y_dist > 5) : 
        while num_centers < num_allowable_centers :
            iter_count += 1
            
            try :
                # Sometimes no circle is detected                
                centers, r_estimated, metric = imfindcircles(dest_img, 
                                                             [min_rad, max_rad],
                                                             sensitivity = sensitivity)
                dest_circles = np.concatenate((centers, r_estimated[:,np.newaxis]), axis = 0).T
                dest_circles = np.squeeze(dest_circles)
            except : 
                dest_circles = []

            num_centers = len(dest_circles)

            sensitivity += 0.1

            if iter_count > 5 : 
                print('After 200 iteration, 4 circles are not detected.')
                return [0., 0., 0.], [[1, 1, 1], [1, 1, 1]]
            
        dest_circles = find_valid_dest_circles(dest_circles)
        dest_circles = dest_circles[dest_circles[:,0].argsort()] # 이에 대해 sort
        
        x = dest_circles[:, 0] # extract x value of circles - get col vector
        y = dest_circles[:, 1] # extract y value of circles - get col vector

        x_dist= abs((x[0] + x[3]) - (x[1] + x[2]))
        y_dist = abs((y[0] + y[3]) - (y[1] + y[2]))
        
        num_allowable_centers += 1

    displacement = homography_transformation(src_circles, dest_circles)

    return displacement, dest_circles

