#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.

import itertools
import os.path
import numpy as np
import cv2
from .utils import findProjectiveTransform

## Function with Path
def convert_by_path(dest_path, src_path):
    dest_img = cv2.imread(dest_path)
    dest_img = cv2.medianBlur(dest_img, 5)

    src_img = cv2.imread(src_path)
    src_img = cv2.medianBlur(src_img, 5)

    convert_by_img(dest_img, src_img)

## Function with ndarray(img)
def convert_by_img(dest_img,
                   src_img,   
                   **input_params
                  ):
    
    param1 = input_params.get("param1",200) 
    param2 = input_params.get("param2",25)
    p_length = input_params.get("p_length",50)
    min_rad = input_params.get("min_rad",70)
    max_rad = input_params.get("max_rad",100)
    
    ## Parameters
    # constant for weight matrix
    # set max, min radius of finding circles

    ## 흑백화면 생성 - For cv2.HoughCircles()
    grey_dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)
    grey_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
#     _, grey_dest_img = cv2.threshold(grey_dest_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     _, grey_src_img = cv2.threshold(grey_src_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    

    ## 이미지 변환 -> dim : (3, 4, 1)
    ## 검출률 변경을 위해선 param2 변경

    src_circles = cv2.HoughCircles(grey_src_img, 
                                   cv2.HOUGH_GRADIENT,
                                   1, 
                                   max_rad*2, 
                                   param1=param1,
                                   param2=param2,
                                   minRadius=min_rad,
                                   maxRadius=max_rad)[0]
    
    iter_count = 1
    num_centers = 0

    while num_centers < 4:
        iter_count += 1
        
        try :
            dest_circles = cv2.HoughCircles(grey_dest_img, 
                                            cv2.HOUGH_GRADIENT, 
                                            1,
                                            max_rad*2,
                                            param1=param1, 
                                            param2=param2, 
                                            minRadius=min_rad,
                                            maxRadius=max_rad)[0] 
            
        except:
            dest_circles = []
            

        num_centers = len(dest_circles)

        param2 -= 1
                
        if iter_count > 200 : 
            print('After 200 iteration, 4 circles are not detected.')
            return [0., 0., 0.]
        
#     if len(dest_circles) > 4 : 
#         print(dest_circles)

    ## src에 대한 호모그래피 좌표변환
    # [[x1, y1], [x2, y2], ...]
    src_matrix = np.array([
        [src_circles[0][0], src_circles[0][1]],
        [src_circles[1][0], src_circles[1][1]],
        [src_circles[2][0], src_circles[2][1]],
        [src_circles[3][0], src_circles[3][1]]
    ])

    # x value를 기준으로 sort 진행
    src_matrix = src_matrix[src_matrix[:,0].argsort()]

    # src_matrix를 Transpose : [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    # src_matrix = src_matrix.T

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
    dist_4 = dist_3 / dist_3[2]

    dist_4 = np.dot(h_matrix, dest_vec4)
    dist_4 = dist_4 / dist_4[2]

    ## 결과 도출 - result
    result = (dist_1 + dist_2 + dist_3 + dist_4) / 4 - np.array([[p_length/4], [p_length/4], [1]])

    return result[:2]
    