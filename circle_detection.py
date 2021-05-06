import cv2 
import os 
import json
import csv
import io
import pickle
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from datetime import datetime
from sscv import imwrite

from glob import glob 
from datetime import datetime
from pathlib import Path
from functools import partial
from p_tqdm import p_map
from tqdm import tqdm 

from module.utils import imread, imfindcircles 
from module.utils import circle_detection_multi_thread, find_valid_dest_circles, adjust_gamma
from module.utils import cvtClcToDict, saveCDResult, tmpListToDict, dirListToDict, getDirList, snsInfoToDict, cvtDataForAPCA, plotData
from module.disp_measure import homography_transformation, adaptiveThreshold_3ch
from module.apca import run_apca_cas 


def do_CircleDetection(img_file_path, params): 

    from module.utils import imread, imfindcircles 

    """
    Args 
        img_file_path 
        params 

    """
    img = imread(img_file_path)
    sensitivity = params['sensitivity']
    circles = []
    try : 
        while len(circles) < 4 :
            centers, r_estimated, _ = imfindcircles(img, 
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
                return circles
    except : 
        circles = 'circle detection is failed .'
        return circles

    circles = find_valid_dest_circles(circles).tolist()

    return circles

def do_CircleDetectionByFolder(date_folder, param_config):
    """
    Args : 
        date_folder 
        param_config
    """
    
    # get current time 
    start_time = datetime.now()
    st_string = start_time.strftime("%Y-%d-%m %H-%M-%S")
    
    # sensor number is 
    # The sensor number can be found in 4 steps from the end of directories .
    sensor_num = date_folder.split('/')[-4]
    
    # folder names 
    log_folder = osp.join(date_folder, 'logs')
    currentime_folder= osp.join(log_folder, st_string)
    circle_detection_folder = osp.join(currentime_folder, 'circle_detection_result_images')
    circles_pickle_filename = osp.join(currentime_folder, 'circle_detection_results.pickle')
    
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    Path(currentime_folder).mkdir(parents=True, exist_ok=True)
    Path(circle_detection_folder).mkdir(parents=True, exist_ok=True)
    
    img_list = glob(osp.join(date_folder, '*.jpg'))
    
    params = param_config[sensor_num].copy()
    info_filename = osp.join(currentime_folder, 'info.json')
    
    num_succeeded_detection = 0
    circles_dict = dict()
    for img_file_path in img_list : 
        
        img_file_base = osp.basename(img_file_path)
        img = imread(img_file_path)
        
        circles = do_CircleDetection(img_file_path, params)
        
        circles_dict[img_file_base] = circles
        
        # do_CircleDetection returns a list of circles when circle detection has no problem
        if str(type(circles)) == "<class 'list'>": 
            # Convert the circle parameters a, b and r to integers. 
            circles = np.uint16(np.around(circles)) 
            if circles.ndim ==1 : 
                circles = circles[np.newaxis, :]
            for pt in circles: 
                a, b, r = pt[0], pt[1], pt[2]   
                # Draw the circumference of the circle. 
                cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 

            imwrite(osp.join(circle_detection_folder, img_file_base), img)
            num_succeeded_detection +=1
        else:
            json_filename = img_file_base.split('.')[0] + '.json'
            print("circles is not detected")
            with open(osp.join(circle_detection_folder, json_filename) , 'w') as outfile:
                circle_detection_error_message = {'Error' : 
                                                  'Circle detection for {} is failed.'.format(img_file_base)}
                json.dump(circle_detection_error_message, outfile)
    
    with open(circles_pickle_filename , 'wb') as outfile:
        pickle.dump(circles_dict, outfile)
        
    with open(info_filename , 'w') as outfile:
        circle_detection_information = dict()
        end_time = datetime.now()
        et_string = end_time.strftime("%Y-%d-%m %H-%M-%S")
        num_imgs_in_list = len(img_list)
        circle_detection_information['num_succeeded_detection'] = '{}/{} succeeded'.format(num_succeeded_detection, num_imgs_in_list)
        circle_detection_information['params'] = params
        circle_detection_information['start time'] = st_string
        circle_detection_information['end time'] = et_string
        circle_detection_information['Elapsed time'] = '{}'.format(end_time - start_time )
        json.dump(circle_detection_information, outfile)
    
    
            
SNS_INFO = pd.read_csv('sensor_info.csv')

# set dataset folder directory 
data_root = '/media/sss/Seagate Backup Plus Drive/NIA_Smart_Monitoring/folderized_data'

# Sensor type 
sns_type = 'Img'

# create image sensor list from folder 
# loop along folders under Img 
# img_sns_list = []
# for folder in glob(osp.join(data_root, sns_type, '*')): 
#     img_sns_list.append(osp.basename(folder))

# img_sns_list.sort()
"""
 '002', '005', '008', '011', '014', '017', "023", "026",
              '029', '032','035','038','041','044','047','050','062',
"""
img_sns_list = ['065','068',
              '071','074','077','090','093','094','097','098','103','104','107',
              '108','111','112','115','116','119','120','123','124','127','128',
              '131','132','135','136','139','140','143','144','147','148','151',
              '152','155','156','159','160','163','164','167','168','171','172',
              '175','176','179','180','183','184','187','188','191','192','195',
              '196','199','200','203','204','207','208','211','212']
# search imgs in subdirectories 
img_folder_list = []
for sns_num in tqdm(img_sns_list) : 

    img_list = glob(osp.join(data_root, sns_type, sns_num, '*', '*', '*'))
    # set this one as parameter    
    filtered_list_by_year =  [item for item in img_list if int(item.split('/')[-3]) >= 2021]
#     filtered_list_by_month =  [item for item in filtered_list_by_year if int(item.split('/')[-2]) >= 9]
#     filtered_list_by_date =  [item for item in filtered_list_by_month if int(item.split('/')[-1]) >= 1]
    img_folder_list.append(filtered_list_by_year)
    
    
with open('params.json') as circle_detection_params_json : 
    param_config = json.load(circle_detection_params_json)
    

for img_folder in img_folder_list[1:]:
    start_time = datetime.now()
    sensor_num = img_folder[0].split('/')[-4]
    num_subfolder = len(img_folder)
    num_images = 0
    for img_path in img_folder:
        img_list = glob(osp.join(img_path, '*.jpg'))
        num_images += len(img_list)
    print('No.{} sensor is under circle detection ----------'.format(sensor_num))
    print('{} images in {} subfolders will be proceeded. ----------'.format(num_images, num_subfolder))
    print('Start time is {}'.format(start_time))

    # for img_folder_by_date in tqdm(img_folder): 
    #     do_CircleDetectionByFolder(img_folder_by_date, param_config=param_config)
    
    p_map(partial(do_CircleDetectionByFolder, param_config=param_config), img_folder, num_cpus = 2)
    
    end_time = datetime.now()
    
    print('Circle detection for NO. {} sensor is complete.------------'.format(sensor_num))
    print('End time is {}'.format(
        end_time))
    print('Elapsed time is {}'.format(end_time-start_time))
    