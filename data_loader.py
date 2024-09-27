import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
import os
import re
import ast
import scipy.io as sio
from scipy.io import loadmat, savemat
import mat73
import wfdb


import warnings
warnings.filterwarnings("ignore")

import rhdecgPre_process as prep
import compute_temporal_feat as tempfeat


# read all data----- split into K-folds based on subjects
file_indx_list =[]
idx_folds = []
num_fold = 1
path = 'D:/path-of-the-Experimental-dataset/'
dataset_loc1 = path+'normal_data' # Folder of Healthy controls
dataset_loc2 = path+'rhd_data' # Folder of RHD subjects
plt.rcParams['figure.figsize'] = [30,5]

def load_data(d_type_in):

    file_indx_list = []
    filelist = []       

    # store all the file names in this list
    if d_type_in == 'normal':
        data_path = dataset_loc1
    else:
        data_path = dataset_loc2

    for root, dirs, files in os.walk(data_path):
        for file in files:
          #append the file name to the list
          if(file.endswith(".mat")):
            filelist.append(os.path.join(root,file))
            file_id = [int(s) for s in re.findall("\d+", os.path.join(root,file)) ]
            file_indx_list.append(file_id[1])
    
    ecg_train = []
    ecg_test = []
    rec_list = []
    group_id = []
    secs=30
    fs=500
    # TRAIN-TEST by SUBJECTS
    for name in filelist:
        # get the subject's folder id
        indx = [int(s) for s in re.findall("\d+", name) ]
        # Load the signal from .mat fie
        temp_ecg = loadmat(name)
        temp_ecg = temp_ecg['ECGrecord'].T
        # Arrange in 10sec length and reshape 
        # Read record
        temp_ecg= temp_ecg[:-1,:]

        if d_type_in=='rhd':
            _, temp_ecg_ = prep.clean_data_rhd(name, indx, temp_ecg)
        else:
            _, temp_ecg_ = prep.clean_data_normal(name, indx, temp_ecg)
        if temp_ecg_.shape[0]==0:
            print('Noisy recording of the subject:', [indx[1],indx[4]])
        else:
            # Checking for less than 10s recording : basically one Normal subject
            if temp_ecg_.reshape(-1).shape[0] <fs*10:
                print(f'Recording Less than 10s duration: {temp_ecg_.reshape(-1).shape[0]} {name}')
                  # temp_ecg_ = np.append(temp_ecg_,int(fs*10-len(temp_ecg_)),constant)      
            secs10=10
            # collect in either train or test based on the subject split index
            temp_ecg_ = np.expand_dims(temp_ecg_[:fs*secs10].reshape(-1),axis=0)[:,:int(fs*secs10)]
            ecg_train.append(temp_ecg_)
            rec_list.append([indx[1], temp_ecg_.shape[0]])
            group_id.append(indx[1])

    ecg_trainX = np.concatenate(ecg_train).ravel().reshape(-1,secs10*fs)
    group_id = np.array(group_id)
    if d_type_in=='normal':
        ecg_trainY = np.zeros(ecg_trainX.shape[0])
    else:
        ecg_trainY = np.ones(ecg_trainX.shape[0])
    rec_list = np.array(rec_list)

    return ecg_trainX, ecg_trainY, rec_list, group_id
