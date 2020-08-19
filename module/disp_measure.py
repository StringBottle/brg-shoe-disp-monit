#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This module uses Numpy & OpenCV(opencv-python). Please install before use it.
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
                   param1 = 80,
                   param2 = 50,
                   p_length = 50,
                   min_rad = 70,
                   max_rad = 100                   
                  ):

    ## Parameters
    # constant for weight matrix

    # set max, min radius of finding circles


    ## 흑백화면 생성 - For cv2.HoughCircles()
    grey_dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)
    grey_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    ## 이미지 변환 -> dim : (3, 4, 1)
    ## 검출률 변경을 위해선 param2 변경

    src_circles = cv2.HoughCircles(grey_src_img, 
                                   cv2.HOUGH_GRADIENT,
                                   1, 
                                   20, 
                                   param1=param1,
                                   param2=param2,
                                   minRadius=min_rad,
                                   maxRadius=max_rad)[0]
    
    iter_count = 1
    num_centers = 0
    cnt_direction_change_2 = 0
    sensi_adj_drct_hist = [0]

    while num_centers != 4:
        iter_count += 1
        try:
            dest_circles = cv2.HoughCircles(grey_dest_img, 
                                            cv2.HOUGH_GRADIENT, 
                                            1,
                                            20,
                                            param1=param1, 
                                            param2=param2, 
                                            minRadius=min_rad,
                                            maxRadius=max_rad)[0] 
            
            num_centers = len(dest_circles)
            
        except:
            print('cv2.HoughCircles found no circles, try with another param2')
        
        sensi_adj_drct = num_centers - 4
        
        if sensi_adj_drct < 0 :
            if iter_count < 100: 
                param2 -= 10**-(cnt_direction_change_2-1)
            else : 
                param1 -= 10**-(cnt_direction_change_2-1)
        elif sensi_adj_drct > 0:
            if iter_count < 100: 
                param2 += 10**-(cnt_direction_change_2-1)
            else : 
                param1 += 10**-(cnt_direction_change_2-1)

        sensi_adj_drct_hist.append(sensi_adj_drct)
        if sensi_adj_drct_hist[-1] * sensi_adj_drct_hist[-2] < 0:
            if iter_count < 100 : 
                cnt_direction_change_2 += 1
            else : 
                cnt_direction_change_1 += 1
        if iter_count > 200 : 
            return [0., 0., 0.]

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

    return result

# Deprecated Code!
    # for img_num in range(1, len(img_file_list)+1):
    #     print('No.' + str(img_num) + '. ' + img_file_list[img_num] + ' is under process.')

    #     img_path = img_file_list[img_num]
    #     process_img = cv2.imread(img_path)

    #     iter_count = 1
    #     centers = []
    #     cnt_direction_change = 0
    #     sensi_adj_drct_hist = [0]

    #     ## 검출률 변경을 위해선 param2 변경
    #     process_circle = cv2.HoughCircles(process_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=min_rad, maxRadius=max_rad)[0]
        
    #     sensi_adj_drct = len(centers) - 4

    #     if sensi_adj_drct > 0:
    #         sensitivity -= 10**-(cnt_direction_change+2)
    #     elif sensi_adj_drct < 0:
    #         sensitivity += 10**-(cnt_direction_change+2)

    #     sensi_adj_drct_hist.append(sensi_adj_drct)

    #     if sensi_adj_drct_hist[-1] * sensi_adj_drct_hist[-2] < 0:
    #         cnt_direction_change += 1

    #     process_circle[process_circle[:,0].argsort()]
    #     lambda x : [[x[0][i], x[1][i]] for i in range()]
        
    #     avg_of_H = (H_1 + H_2 + H_3 + H_4)/4
    #     result[img_num] = np.subtract(avg_of_H, np.array([[p_length/4], [p_length/4], [p_length/4]]))
    