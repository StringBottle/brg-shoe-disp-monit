# import required libraires 

import shutil
import os.path as osp 
import pysftp

from pathlib import Path
from glob import glob 
from tqdm import tqdm
from p_tqdm import p_map

dataset_dir = '/media/sss/Seagate Backup Plus Drive/NIA_Smart_Monitoring'
file_list =  glob(osp.join(dataset_dir, '2020', 'data', "*"))
folderized_dataset_dir = '/media/sss/Seagate Backup Plus Drive/NIA_Smart_Monitoring/folderized_data'

Hostname = "121.161.206.168"
Username = "root"
Password = "tltjf2019#"

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

dataset_dir_in_sftp = '/home/everythingadt/FSMS_SRV/data'



with pysftp.Connection(host=Hostname, username=Username, password=Password, 
                     cnopts=cnopts, port=9017) as sftp:
    with sftp.cd('..'):
        print('Start reading file list from server.')
        print('Server IP : {}'.format(Hostname))
        print('Access to Directory : {}'.format(dataset_dir_in_sftp))
        filelist = sftp.listdir(dataset_dir_in_sftp)
        print('File list is read successfully.')
        idx = 0
        # get basename file list from filelist 
        # get basename from file list in 'folderized_dataset_dir'
        # remove basenames 
        
        # def download_multiprocessing(filename):
        #     file_name_spilt = filename.split('_')
        #     if (file_name_spilt[0] == 'AES') or (file_name_spilt[0] == 'Img') or (file_name_spilt[0] == 'Tmp') : 
        #         sns_type = file_name_spilt[0] 
        #         sns_num = file_name_spilt[1]
        #         year = file_name_spilt[2][:4]
        #         month = file_name_spilt[2][4:6]
        #         date = file_name_spilt[2][6:8]

        #         save_path = osp.join(folderized_dataset_dir, sns_type, sns_num, year, month, date)

        #         if not osp.exists(osp.join(save_path, filename)) : 
        #             # print('File is already downloaded, so downloading is skipped. File name is ' + file)
        #             # else: 
        #             Path(save_path).mkdir(parents=True, exist_ok=True)
        #             sftp.get(osp.join(dataset_dir_in_sftp, filename), osp.join(save_path,  filename))
        #             print('File Transfer is successful, file name is ' + filename)


        # p_map(download_multiprocessing, filelist)
        

        for file in tqdm(filelist) : 
        #     file_basename = osp.basename(file)
        #     print(file)
            file_name_spilt = file.split('_')
            if (file_name_spilt[0] == 'AES') or (file_name_spilt[0] == 'Img') or (file_name_spilt[0] == 'Tmp') : 
                sns_type = file_name_spilt[0] 
                sns_num = file_name_spilt[1]
                year = file_name_spilt[2][:4]
                month = file_name_spilt[2][4:6]
                date = file_name_spilt[2][6:8]

                save_path = osp.join(folderized_dataset_dir, sns_type, sns_num, year, month, date)

                if not osp.exists(osp.join(save_path, file)) : 
                    # print('File is already downloaded, so downloading is skipped. File name is ' + file)
                # else: 
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    sftp.get(osp.join(dataset_dir_in_sftp, file), osp.join(save_path,  file))
                    print('File Transfer is successful, file name is ' + file)
                    
            idx = idx + 1    

        print('Downloading is complete.')

