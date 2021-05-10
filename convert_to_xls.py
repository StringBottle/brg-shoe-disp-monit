import io
import pickle

import numpy as np
import os.path as osp 
import pandas as pd

from glob import glob 
from pathlib import Path
from tqdm import tqdm
from module.disp_measure import homography_transformation
from module.utils import str2array


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

def search_directories_by_sensor(data_root, sns_type, sns_list=None) : 
    
    """
    Args : 
        data_root
        sns_type
        sns_list

    Returns : 
        directories_by_sensor

    """
    directories_by_sensor = [] 

    if sns_list : 
        for sns_num in tqdm(sns_list, desc='Search Directories by Sensor : '): 
            directories = glob(osp.join(data_root, sns_type, sns_num, '*', '*', '*'))
            directories_by_sensor.append(directories)
        
    else : 
        sns_list = glob(osp.join(data_root, sns_type, '*'))
        for sns_num_ in tqdm(sns_list, desc='Search Directories by Sensor : '):
            sns_num = osp.basename(sns_num_)
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


def get_src_circles(sns_num):

    """
    Args : 
        sns_num (int or float) : sensor number 

    Returns : 
        src_circles (np.array) : center coordinates of source circles 
        
    """
    # default source image number 
    year = '2020'
    month = '10'
    date = '14'
    time = '16'

    # exceptions     
    if sns_num == '011':
        time = '01'
    elif sns_num == '041':
        time = '16'
    elif sns_num == '135':
        time = '07'
    elif sns_num == '071':
        time = '11'
    elif sns_num == '155':
        time = '23'
    
    src_circles_folder = osp.join(data_root, sns_type, sns_num, year, month, date) 

    latest_log = get_latest_log_folder(src_circles_folder)

    circles_pickle_filename = osp.join(latest_log, 'circle_detection_results.pickle')

    with open(circles_pickle_filename , 'rb') as readfile:
        src_circles_dict = pickle.load(readfile)

    img_key = 'Img_{}_{}{}{}_{}0100.jpg'.format(sns_num, year, month, date, time)

    return src_circles_dict[img_key]


def get_latest_log_folder(date_folder):

    """
    Args :
        date_folder (str) : directory of date_folder 

    Return : 
        latest_log (str) : latest log folder name 

    """

    log_folder = osp.join(date_folder, 'logs')
    exist_log_folder = Path(log_folder).exists()

    if exist_log_folder : 
        log_folders_by_time = glob(osp.join(log_folder, '*'))
    else : 
        print("There is no log folder exists under {}.".format(date_folder))
        return None

    if len(log_folders_by_time) > 1 : 
        log_folders_by_time.sort
        latest_log = log_folders_by_time[-1]
    else: 
        latest_log = log_folders_by_time[0]

    return latest_log


def get_latest_circle_pkl(log_folders_by_time):
    
    for latest_log in reversed(log_folders_by_time):
        circles_pickle_filename = osp.join(latest_log, 'circle_detection_results.pickle')
        if Path(circles_pickle_filename).exists():
            return circles_pickle_filename
    return None


def collect_disp_info(img_folder_list):

    """
    Args: 
        img_folder_list (list) : directory list of image folders to scan 

    Return: 
        disp_info_dict (dict) : collected displacement dict 
    """

    disp_info_dict = {}

    for img_folder in tqdm(img_folder_list, desc="Collecting displacement data : "): 
        sns_num = img_folder[0].split('/')[-4]
        disp_info = pd.DataFrame(columns=['datetime', 'circles','x_disp','y_disp'])

        for date_folder in img_folder : 
            
            # latest_log = get_latest_log_folder(date_folder)
            log_folder = osp.join(date_folder, 'logs')
            exist_log_folder = Path(log_folder).exists()

            if exist_log_folder : 
                log_folders_by_time = glob(osp.join(log_folder, '*'))
            else : 
                print("There is no log folder exists under {}.".format(date_folder))
                continue

            log_folders_by_time.sort

            for latest_log in reversed(log_folders_by_time):
                circles_pickle_filename = osp.join(latest_log, 'circle_detection_results.pickle')
                if Path(circles_pickle_filename).exists():
                    break

            if not Path(circles_pickle_filename).exists():
                print("There is no circle_detection_results exists under {}.".format(latest_log))
                continue


            with open(circles_pickle_filename , 'rb') as readfile:
                circles_dict = pickle.load(readfile)
            
            for img_name, circles in circles_dict.items() : 

                _, _ , year_month_date, hh_mm_ss = img_name.split('_')
                
                if not isinstance(circles, str) : 
                    src_circles = np.asarray(get_src_circles(sns_num))
                    circles = np.asarray(circles)
                    disp_result = homography_transformation(src_circles[:, :2], circles[:, :2])
                    x_disp, y_disp = float(disp_result[0]), float(disp_result[1])

                else : 
                    x_disp, y_disp = None, None 

                disp_info = disp_info.append({'datetime' : str(year_month_date) + str(hh_mm_ss[:2]),
                                            'circles' : circles,
                                            'x_disp': x_disp,
                                            'y_disp': y_disp }, ignore_index = True)
                    
        disp_info_dict[sns_num] = disp_info

    return disp_info_dict


def collect_temp_info(tmp_folder_list): 

    tmp_dict = {}
    for tmp_folder in tqdm(tmp_folder_list, desc='Collecting Temperature Data : '): 
        
        sns_num = tmp_folder[0].split('/')[-4]
        tmp_info = pd.DataFrame(columns=['datetime', 'temp'])
        
        for date_folder in tmp_folder: 
            
            tmp_list = glob(osp.join(date_folder, '*.tpr'))
            for tmp_file in tmp_list : 
                tmp_measured = None
                f = io.open(tmp_file, mode="r", encoding="utf-8")
                txt = f.read()
                if txt :
                    tmp_measured = str2array(txt)[1]

                tmp_basename = osp.basename(tmp_file)
                # sns_type = tmp_basename[:3]
                # sns_id = tmp_basename[4:7]
                date = tmp_basename[8:16]
                time_ = tmp_basename[17:23]
                
                tmp_info = tmp_info.append({'datetime' : str(date) + str(time_[:2]) , 
                                            'temp' : tmp_measured} , ignore_index = True)
                
        tmp_dict[sns_num] = tmp_info

    return tmp_dict


def main(): 

    img_sns_dirs = search_directories_by_sensor(data_root, sns_type, img_sns_list )

    tmp_sns_dirs = search_directories_by_sensor(data_root, 'Tmp')

    filtered_img_sns_dirs = filter_directories_by_date(img_sns_dirs, 2020, 9, 1)
    filtered_tmp_sns_dirs = filter_directories_by_date(tmp_sns_dirs, 2020, 9, 1)

    # create excel format dataframe from sns information
    with open('data_tmp.pkl', "rb") as f:
        sns_dict = pickle.load(f)

    sns_info_df = create_sns_info_df(sns_dict)
    
    tmp_info_dict = collect_temp_info(filtered_tmp_sns_dirs)
    disp_info_dict = collect_disp_info(filtered_img_sns_dirs)
    
    writer = pd.ExcelWriter(xls_filename, engine='xlsxwriter')

    sns_info_df.to_excel(writer, sheet_name= 'snsr_info')

    for sns_num, disp_data in disp_info_dict.items():
        disp_data.to_excel(writer, sheet_name= sns_num)

    for sns_num, tmp_data in tmp_info_dict.items():
        tmp_data.to_excel(writer, sheet_name= sns_num)

    writer.save()


if __name__ == "__main__" : 

    main()

    

