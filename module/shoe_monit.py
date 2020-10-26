import cv2 
import json
import io
import numpy as np
import os.path as osp 
import matplotlib.pyplot as plt 
import pandas as pd

from glob import glob
from functools import partial
from p_tqdm import p_map
from tqdm import tqdm 
from module.utils import imread, imfindcircles, findProjectiveTransform, str2array, circle_detection_multi_thread
from module.disp_measure import displacement_measure, homography_transformation, find_valid_dest_circles
from module.apca import run_apca_cas 



class ShoeMonitObj():
    
    def __init__(self, param_filename, anal_table_filename, snsr_info_filename, 
                 dataset_folder):
        
        # Import circle detection and APCA parameters 
        with open(param_filename) as param_config_json : 
            self.param_config = json.load(param_config_json)

        '''
        1. Check if any images where circles detection is not performed.
        & Run imfindcircles for recently added images and save to csv 
        '''
        self.anal_table_filename = anal_table_filename

        if anal_table_filename : 
            self.anal_table = pd.read_csv(anal_table_filename)
        else : 
            self.anal_table = pd.DataFrame(columns=['image_name','centers_of_detected_circles',
                                                   'num_centers','x_displacement','y_displacement'])
        self.snsr_info = pd.read_csv(snsr_info_filename)
        self.dataset_folder = dataset_folder
        
        self.img_snsr_list = self.snsr_info[(self.snsr_info['Sensor Type'] == 'Img')]['Sensor ID']
        self.tmp_snsr_list = self.snsr_info[(self.snsr_info['Sensor Type'] == 'Tmp')]['Sensor ID']
    
    def data_list_to_tmp_data(self, time_list, tmp_data_list): 

        # make this section into two groups 
        # 1. read tmp data list 
        # 2. read tmp data from tmp data list 
        # 3. add tmp location option? 
        tmps_at_tm_list = []
        for tm in time_list: 
            # sensor displacement 
            tmp_list_match = []
            for name in tmp_data_list:
                if tm in name:
                    tmp_list_match.append(name)

            tmp_values = []
            for tmp_file in tmp_list_match: 
                f = io.open(tmp_file, mode="r", encoding="utf-8")
                txt = f.read()
                if txt :
                    tmp_measured = str2array(txt)[1]
                    tmp_values.append(tmp_measured)

            if tmp_values:
                tmps_at_tm_list.append(np.nanmean(tmp_values))
            else:
                tmps_at_tm_list.append(22)
                    
        return tmps_at_tm_list

    
    def get_data_list(self, snsr_type, slab_num = None, snsr_id = None, snsr_loc = None, anal_range = [20200901, 90200901]):
        
        # To-do:
        # Add filtering option for block group, sensor_num, etc
        # return 
        # data_list = filename_list 
        
#         if snsr_type == 'Img': 
#             snsr_list = self.img_snsr_list
#         elif snsr_type == 'Tmp' :
#             snsr_list = self.tmp_snsr_list

        if snsr_id : 
            snsr_list = self.snsr_info[(self.snsr_info['Sensor ID'] == snsr_id) ]['Sensor ID']
            
        elif slab_num : 
            snsr_list = self.snsr_info[(self.snsr_info['Sensor Type'] == snsr_type) &(self.snsr_info['Slab No'] == slab_num)]['Sensor ID']
            
        elif snsr_loc : 
            snsr_list = self.snsr_info[(self.snsr_info['Sensor Type'] == snsr_type) & (self.snsr_info['Slab No'] == slab_num) & (self.snsr_info['Sensor Location'] == snsr_loc)]['Sensor ID']
            
        else : 
            snsr_list = self.snsr_info[(self.snsr_info['Sensor Type'] == snsr_type) &(self.snsr_info['Slab No'] == slab_num)]['Sensor ID']
        
            
            
        data_list = []
        snsr_data_list = []
            
        # get filename by sensor list
        for snsr_id in snsr_list:
        # check installation year 

            install_year = self.snsr_info[(self.snsr_info['Sensor ID'] == snsr_id)]['Sensor Installation Year'].values[0]
            data_folder_name = osp.join(self.dataset_folder, str(install_year), 'data')
            snsr_data_list += glob(osp.join(data_folder_name, snsr_type + '_' + str(snsr_id).zfill(3)+'*.jpg'))
        
        # filter list by date 
        for snsr_data in tqdm(snsr_data_list): 
            snsr_data_date = int(osp.basename(snsr_data)[8:16])

            if (snsr_data_date >= anal_range[0]) and (snsr_data_date <= anal_range[1]):
                
                data_list.append(snsr_data)
                
        data_list.sort()
       
        return data_list
        
        
    
    def crcl_det(self, anal_range=[20200901, 90200901], apca_group = None,
                overwrite_anal_table = True):
        
        
        img_list_for_cd = self.get_data_list(anal_range=anal_range, snsr_type='Img')
        run_circle_detection_list = []

        # create a list of imgs to which re-run circle detection 
        for img_file in img_list_for_cd : 
            img_name = osp.basename(img_file)
            num_image_in_the_cvs = len(np.where(self.anal_table['image_name']== img_name)[0])
            if num_image_in_the_cvs : 
                img_idx = int(np.where(self.anal_table['image_name']== img_name)[0]) 
                num_circles = self.anal_table['num_centers'][img_idx] 

            else : 
                run_circle_detection_list.append(img_file)

        if run_circle_detection_list : 
            run_circles_list = p_map(partial(circle_detection_multi_thread, param_config=self.param_config), run_circle_detection_list)
            

            for img_name, circles in zip(run_circle_detection_list, run_circles_list):
                img_basename = osp.basename(img_name)
                sensor_num = str(img_basename[4:7])
                params = self.param_config[sensor_num]
                new_row = [img_basename , circles,  len(circles), float('NaN'),  float('NaN'),]
                s = pd.Series(new_row, index=self.anal_table.columns )
                self.anal_table = self.anal_table.append(s, ignore_index=True)
                
            if overwrite_anal_table : 

                self.anal_table.sort_values(by = 'image_name')
                self.anal_table.to_csv(self.anal_table_filename, index=False)
            
    def disp_measure(self):
        '''
        2. Convert circle center coord to displacement 
        '''
        ## This code assumes that displace measurement has been done yet. 
        ## Need to edit this code to work when previous displacement measurement results exist.

        x_displacement_list = []
        y_displacement_list = []

        # since displacement measurement doesn't take so long, 
        # This code will be written just to re-write displacement column everytime 

        for num, img_name in enumerate(self.anal_table['image_name']) :
            try :
                sric_idx = int(np.where(self.anal_table['image_name']== self.param_config[img_name[4:7]]['src_img'])[0]) 
                src_circles = str2array(self.anal_table['centers_of_detected_circles'][sric_idx])
                dest_circles = str2array(self.anal_table['centers_of_detected_circles'][num])

                displacement = homography_transformation(src_circles, dest_circles)
                x_displacement_list.append(displacement[0][0])
                y_displacement_list.append(displacement[1][0])

            except IndexError as error:
            # Output expected IndexErrors.
                print(img_name, error)
                x_displacement_list.append([0])
                y_displacement_list.append([0])
            except Exception as exception:
                # Output unexpected Exceptions.
                print(img_name, exception)
                x_displacement_list.append([0])
                y_displacement_list.append([0])

        columns_to_overwrite = ['x_displacement', 'y_displacement']
        self.anal_table.drop(labels=columns_to_overwrite, axis="columns", inplace=True)


        self.anal_table['x_displacement'] = x_displacement_list
        self.anal_table['y_displacement'] = y_displacement_list

        self.anal_table.to_csv('analysis_table.csv',index=False)
        
    def anomal_det(self, training_range = 100, anal_range = [20200901, 90200901],
                   slab_num = None, snsr_id = None, snsr_loc = None,
                   anomal_detector = 'apca'): 
   
        # anomal_det_option 
        # 1. anomal detector / currently only support apca 
        # 2. anomal detect group 
            # a) tmp + disp of a shoe ,
            # b) tmp + disp of shoes on one side of slab
            # c) tmp + disp of shoes on both side of slab 
            
            # To implement 2., 
            # disp_data_group 
            # 1. sensor name 
            # 2. slab name with starting or ending point 
            # 3. only slab name 
            # tmp data is determined by the sensor location 
        # training data range / test data range 
            
        # get tmp data list 
        
        tmp_list_for_cd = self.get_data_list(anal_range=anal_range, 
                                             slab_num = None, snsr_id = None,
                                             snsr_loc = None, 
                                             snsr_type='Tmp')
        
        time_list = []
        for img_name in self.anal_table['image_name']:
            if str(self.img_snsr_list.values[0]).zfill(3) in img_name :
                time_list.append(img_name[8:-8])

        op={}
        
        
        tmps_at_tm_list = self.data_list_to_tmp_data(tmp_list_for_cd, time_list)

        # 2. get temperature data in op['t']
            # read temperature data according to the excel file 

        # 3. get displacement data in op['f']

        op['t'] = np.asarray(tmps_at_tm_list)[:, np.newaxis].T  
        
        # adjust displacement direction using sensor excel file 
        
        # get img name using filter 
        # filter 1. sensor_direction 
        # filter 2. anal_range 
        # get disp value using disp_value 
        disp_at_tm_list = []
        for sensor_id in self.img_snsr_list :
            sensor_direction = self.snsr_info[(self.snsr_info['Sensor ID'] == sensor_id)]['Displacement Direction'].values[0]
            if sensor_direction != 'FALSE' :
                disp_list_match = []
                for tm in time_list: 
            #         print('Img_' + str(sensor_id).zfill(3) +'_' + tm + '0100.jpg')
                    img_basename = 'Img_' + str(sensor_id).zfill(3) +'_' + tm + '0100.jpg'
                    disp_value = self.anal_table[(self.anal_table['image_name'] == img_basename)]['x_displacement'].values
                    if len(disp_value) > 0: 
                        if sensor_direction == 'Forward':
                            disp_list_match.append(-disp_value[0])
                        else : 
                            disp_list_match.append(disp_value[0])
                    else : 
                        disp_list_match.append(0)
                disp_at_tm_list.append(disp_list_match)

        disp_at_tm_array = np.asarray(disp_at_tm_list).T
        op['f'] = disp_at_tm_array
        op['IND_x0'] = np.arange(0, 100)
        op['IND_x1'] = np.arange(100, disp_at_tm_array.shape[0])
        # 4. run apca 
        
        
        X1, PC, Q = run_apca_cas(op)
        # add anomal info to op 
        return X1, PC, Q 
        
        # 5. Plot the result 
        
        
    def plot(): 
        fig = plt.figure(figsize=(12, 8))

        x0 = op['IND_x0']
        x1 = op['IND_x1']
        plt.plot(x0, Q['Qdist'][x0], 'bo', linewidth=2, markersize=8)
        plt.plot(x1, Q['Qdist'][x1], 'v', linewidth=2, markersize=8)

        alpha = 0.95
        SD_type = 'Z-score'
        PCA_par = {}
        PCA_par['n_stall'] = 10;

        IND_abnor = np.argwhere(PC['Updated']==0)
        IND_nor = np.argwhere(PC['Updated']==1)
        plt.plot(IND_abnor,Q['Qdist'][IND_abnor],'rs',linewidth=2, markersize=8) # Outlier and faulty samples
        plt.plot(IND_nor,Q['distcrit'][IND_nor][:,:, 0],'k-',linewidth=5, markersize=8) # Outlier and faulty samples
        plt.ylabel('Q-statistics (SPE)',fontsize=20)
        plt.title('Alpha='+str(alpha)+str(SD_type)+'# stall:'+str(PCA_par['n_stall']),fontsize=15,fontweight='bold')
        plt.legend(fontsize='x-large')
        plt.show





        



        

        
     


