#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.

import itertools
import os.path
import numpy as np
import cv2
from itertools import combinations 
from .utils import findProjectiveTransform

## Function that find four valid circles in dest_circles
def find_valid_dest_circles(dest_circles):
    ## I. 상위 4개 원을 x value를 기준으로 sort 진행
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
    if x_dist < 10.0 and y_dist < 10.0: # nice case
        return dest_circles[0:4]
    
    ## III. 탐지된 원들 내에서 4개의 원을 추출하는 모든 경우의 수에 따라 원의 집합을 생성
    ## combination으로 계산되어도 문제 없습니다. 
    comb = combinations(dest_circles, 4) # [[[x1, y1, r1], [x2, y2, r2], ...], [4], [4], [4], ...]
    # IV. 각 원의 집합들에 대하여 2번 과정에 대한 연산을 수행
    dist_list = []
    elem_list = []
    for elem in comb:
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
        elem_list.append(elem)
    
    # V. 4번 연산을 수행하여 가장 낮은 값을 갖는 집합을 반환 
    min_idx = np.argmin(dist_list)
    dest_circles_with_min_dist = elem_list[min_idx]
    
    return dest_circles_with_min_dist

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
    
    # To do : Check if binarization option is needed
    _, grey_dest_img = cv2.threshold(grey_dest_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     _, grey_src_img = cv2.threshold(grey_src_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    grey_dest_img = cv2.morphologyEx(grey_dest_img, cv2.MORPH_GRADIENT, kernel)

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
    x_dist = 11
    y_dist = 11
    num_allowable_centers = 4

    while x_dist > 5 or y_dist > 5 : 
        while num_centers < num_allowable_centers :
            iter_count += 1
            
            try : # Sometimes no circle is detected
                dest_circles = cv2.HoughCircles(grey_dest_img, 
                                                cv2.HOUGH_GRADIENT, 
                                                1,
                                                max_rad*2,
                                                param1=param1, 
                                                param2=param2, 
                                                minRadius=min_rad,
                                                maxRadius=max_rad)[0] 
            except : 
                dest_circles = []

            num_centers = len(dest_circles)

            param2 -= 0.5

            if iter_count > 200 : 
                print('After 200 iteration, 4 circles are not detected.')
                return [0., 0., 0.], [[1, 1, 1], [1, 1, 1]]
            
        dest_circles = find_valid_dest_circles(dest_circles)
        dest_circles = dest_circles[dest_circles[:,0].argsort()] # 이에 대해 sort
        
        x = dest_circles[:, 0] # extract x value of circles - get col vector
        y = dest_circles[:, 1] # extract y value of circles - get col vector

        x_dist= abs((x[0] + x[3]) - (x[1] + x[2]))
        y_dist = abs((y[0] + y[3]) - (y[1] + y[2]))
        
        num_allowable_centers += 1

    
    # Modified by beetea
    

    ## Deprecated code since len(dest_circles) == 4 is always true


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

    return result[:2], dest_circles