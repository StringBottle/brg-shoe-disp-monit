
import pickle

import os.path as osp 
import pandas as pd

from glob import glob 
from pathlib import Path
from tqdm import tqdm


data_root = '/media/sss/Seagate Backup Plus Drive/NIA_Smart_Monitoring/folderized_data'

xls_filename = '2021.05.10.xlsx'

sns_type = 'Img'

img_sns_list = ['002', '005', '008', '011', '014', '017', '023', '026',  
              '029', '032','035','038','041','044','047','050','062','065','068',
              '071','074','077','090','093','094','097','098','103','104','107',
              '108','111','112','115','116','119','120','123','124','127','128',
              '131','132','135','136','139','140','143','144','147','148','151',
              '152','155','156','159','160','163','164','167','168','171','172',
              '175','176','179','180','183','184','187','188','191','192','195',
              '196','199','200','203','204','207','208','211','212']


# filter list by date 

# search imgs in subdirectories 

def search_directories_by_sensor(data_root, sns_type, sns_list) : 
    
    """
    Args : 
        data_root
        sns_type
        sns_list

    Returns : 
        directories_by_sensor

    """
    directories_by_sensor = [] 

    for sns_num in tqdm(sns_list, desc='Search Directories by Sensor : '): 
        directories = glob(osp.join(data_root, sns_type, sns_num, '*', '*', '*'))
        directories_by_sensor.append(directories)
    return directories_by_sensor


def filter_directories_by_date(directories_by_sensor, filter_year, filter_month, filter_date):

    """
    Args : 
        directories_by_sensor
        filter_year
        filter_month
        filter_date

    Returns : 
        filtered_list

    """
    
    filtered_list = []
    for directories in directories_by_sensor:
        file_list = []
        for item in directories:
            year = int(item.split('/')[-3])
            month = int(item.split('/')[-2])
            date = int(item.split('/')[-1])
            if (year > filter_year) : 
                file_list.append(item)
            elif (year == filter_year) and (month > filter_month) : 
                file_list.append(item)
            elif (year == filter_year) and (month == filter_month) and (date >= filter_date) : \
                file_list.append(item)

        filtered_list.append(file_list)

    return filtered_list

def create_sns_info_df(sns_dict):

    """
    Args : 
        sns_dict

    Returns : 
        sns_info_df 
    
    """
    data_fr = pd.DataFrame(sns_dict)
    sns_info_df = pd.DataFrame(columns=['snsr_num','type','pier','slab','block',
                                          'sns_dir', 'sns_loc', 'disp_dir', 'install_year'])

    for k, v in  data_fr.items() : 
        sns_info_df = sns_info_df.append({'snsr_num' : str(k) , 
                                        'type' : v['type'],
                                        'pier':v['pier'],
                                        'slab':v['slab'],
                                        'block':v['block'],
                                        'sns_dir':v['sns_dir'],
                                        'sns_loc':v['sns_loc'],
                                        'disp_dir':v['disp_dir'], 
                                        'install_year' : v['install_year']} , ignore_index = True)

    return sns_info_df

def get_latest_log_folder(log_folder):

    """
    Args :
        log_folder (str) : directory of log folder 

    Return : 
        latest_log (str) : latest log folder name 

    """

    log_folders_by_time = glob(osp.join(log_folder, '*'))

    if len(log_folders_by_time) > 1 : 
        log_folders_by_time.sort
        latest_log = log_folders_by_time[-1]
    else: 
        latest_log = log_folders_by_time[0]

    return latest_log

def collect_disp_info(img_folder_list):

    """
    Args: 
        img_folder_list (list) : directory list of image folders to scan 

    Return: 
        disp_info_list (list) : collected displacement list 
    """

    disp_info_list = []

    for img_folder in tqdm(img_folder_list): 
        sns_num = img_folder[0].split('/')[-4]
        disp_info = pd.DataFrame(columns=['datetime', 'circles','x_disp','y_disp'])

        for date_folder in img_folder : 

            log_folder = osp.join(date_folder, 'logs')
            exist_log_folder = Path(log_folder).exists()

            if exist_log_folder : 
                latest_log = get_latest_log_folder(log_folder)

            circles_pickle_filename = osp.join(latest_log, 'circle_detection_results.pickle')
            circle_detection_folder = osp.join(latest_log, 'circle_detection_result_images')
            # displacement_dict_filename = osp.join(latest_log, 'displacement_measurement.json')
            circle_detection_list = glob(osp.join(circle_detection_folder, "*.jpg"))

            with open(circles_pickle_filename , 'rb') as readfile:
                circles_dict = pickle.load(readfile)
            
            for img_name, circles in circles_dict.items() : 
                

            for circle_detection_image in circle_detection_list:
                image_basename = osp.basename(circle_detection_image)
                image_basename_no_ext = image_basename.split('.')[0]
                sns_type, _ , year_month_date, hh_mm_ss = image_basename_no_ext.split('_')
                dest_circles = circles_dict[image_basename]
                x_disp = displacement_dict[image_basename]['x_disp']
                y_disp = displacement_dict[image_basename]['y_disp']

                disp_info = disp_info.append({'datetime' : str(year_month_date) + str(hh_mm_ss[:2]),
                                              'circles' : dest_circles,
                                              'x_disp': x_disp,
                                              'y_disp': y_disp }, ignore_index = True)
    disp_info_list.append(disp_info)    

    return None



def main(): 

    img_sns_dirs = search_directories_by_sensor(data_root, sns_type, img_sns_list )

    filtered_img_sns_dirs = filter_directories_by_date(img_sns_dirs, 2020, 9, 1)

    # create excel format dataframe from sns information
    with open('data_tmp.pkl', "rb") as f:
        sns_dict = pickle.load(f)

    sns_info_df = create_sns_info_df(sns_dict)

    collect_disp_info(filtered_img_sns_dirs)



if __name__ == "__main__" : 

    main()

    

