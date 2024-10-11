'''
@author: Amsalu Tomas
Created on Sun Sept  14 15:07:18 2024

'''

# # Symptomatic RHD classification from ECG signals
# 
# The RHDECG dataset was recorded at referal cardiac clinic,TASH Ethiopia, from patients having symptoms.
# The ECG data was recorded using a single lead Beat2Phone wearable sensor, sampled at 500Hz.
# 
# ## Dataset description
# ```
# Records | Subjects | Class
# ---------------------------------------------
# 142    | 46       | Normal ECG
# 298    | 124      | RHD
# ```
# 
# After cleaning very noisy records we end up with:
# ```
# Records | Subjects | Class
# ---------------------------------------------
# 138    | 45       | Normal ECG
# 291    | 121      | RHD
# ```


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import neurokit2 as nk
import os, random, re, ast, mat73, wfdb, sys, warnings
import scipy.io as sio
from scipy.io import loadmat, savemat

sys.path.append(data_path) 
warnings.filterwarnings("ignore")

import utils as prep
import get_temporal_feat as tempfeat
import model_defs as models
import feat_importance as importance

# ## Load the dataset 
# read all data----- split into K-folds based on subjects
file_indx_list =[]
idx_folds = []
num_fold = 1
dataset_loc1 = path+'normal_data'
dataset_loc2 = path+'rhd_data'

def load_data(d_type_in, start_method=None):

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
    secs=10
    fs=500
    ecg_len=fs*secs
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
            if temp_ecg_.reshape(-1).shape[0] <ecg_len:
                print('Recording Less than 10s duration:', name)
                temp_ecg_ = np.hstack((temp_ecg_))              
            
            # append in either train or test based on the subject split index
            if start_method=='random':
                start=random.randint(0, len(temp_ecg_.reshape(-1)) - ecg_len)
                peaks, _ = nk.ecg_peaks(temp_ecg_.reshape(-1)[start:start+ecg_len], sampling_rate=fs)
                if (nk.ecg_quality(temp_ecg_.reshape(-1)[start:start+ecg_len], sampling_rate=fs,method='zhao2018')=='Excellent') and int(np.sum(peaks)>10):
                    temp_ecg_ = np.expand_dims(temp_ecg_.reshape(-1)[start:start+ecg_len],axis=0)
                else:
                    start=random.randint(0, len(temp_ecg_.reshape(-1)) - ecg_len)
                    if start >=int(temp_ecg_.reshape(-1).shape[0])-ecg_len:
                        print('====== Start index Error at: ', start)               
                        temp_ecg_ = np.expand_dims(temp_ecg_[:ecg_len,:].reshape(-1),axis=0)
                        plt.plot(temp_ecg_)
                    else:
                        temp_ecg_ = np.expand_dims(temp_ecg_[start:start+ecg_len,:].reshape(-1),axis=0)
            
            else:
                temp_ecg_ = np.expand_dims(temp_ecg_.reshape(-1),axis=0)[:,:ecg_len]
            ecg_train.append(temp_ecg_)
            rec_list.append([indx[1], temp_ecg_.shape[0]])
            group_id.append(indx[1])

 
    ecg_trainX = np.concatenate(ecg_train).ravel().reshape(-1,secs*fs)
    group_id = np.array(group_id)
    if d_type_in=='normal':
        ecg_trainY = np.zeros(ecg_trainX.shape[0])
    else:
        ecg_trainY = np.ones(ecg_trainX.shape[0])
    rec_list = np.array(rec_list)

    return ecg_trainX, ecg_trainY, rec_list, group_id

plt.rcParams['figure.figsize']=[40,8]
plt.rcParams['font.size']=30
ecg_NSR, NSR_labels, rec_list_NSR,group_id_NSR = load_data('normal','nrandom')
ecg_RHD, RHD_labels, rec_list_RHD,group_id_RHD = load_data('rhd','nrandom')       
print('Normal_RHDECG classes: ',ecg_NSR.shape)
print('RHD_RHDECG classes: ',ecg_RHD.shape)


#%% Filter the signals [1 - 100]
 
# Each 10s record was filtered using Butterworth bandpass with low and high cut-off frequencies `[0.5, 100]Hz`. Subsequently, notch filter at 50Hz was applied. 
# For our experiments we downsampled the signal to 250Hz.
# A record is then normalized to zero mean and unit variance.

######## Filter the recordings #############
from scipy import stats, signal
def filter_dataset(temp_ecg_,rate=500,desired_fs=500,secs=10):
    data_filt=[]
    for j in range(len(temp_ecg_)):
        # Bandpass the signals, and antialias 
        b, a = signal.butter(N=5, Wn=[1,100], btype='bandpass', fs=rate, analog=False)
        rec_filt = signal.filtfilt(b, a, temp_ecg_[j].reshape(-1))
        # Remove PLI noise
        b, a = signal.iirnotch(50, Q = 40, fs = rate)
        rec_filt = signal.filtfilt(b, a, rec_filt)

        # Downsample if required
        resampled_sig = nk.signal_resample(rec_filt.reshape(-1), method="resample_method", sampling_rate=rate, desired_sampling_rate=desired_fs)
        data_filt.append(resampled_sig)
    return data_filt

# Call the filtering function for each record
X_NSR_filt = filter_dataset(ecg_NSR,500,500,secs=10)
X_NSR_filt = np.concatenate(X_NSR_filt).ravel().reshape(-1,5000)
print('Filtered NSR data shape (X, Y):', (X_NSR_filt.shape, NSR_labels.shape))
X_RHD_filt = filter_dataset(ecg_RHD,500,500,secs=10)
X_RHD_filt = np.concatenate(X_RHD_filt).ravel().reshape(-1,5000)
print('Filtered RHD data shape (X, Y): ', (X_RHD_filt.shape,RHD_labels.shape))

# save the filtered signals for further analysis
savemat(path+'NSR_X.mat',{'ECGRecord': X_NSR_filt})
savemat(path+'RHD_X.mat',{'ECGRecord': X_RHD_filt})
savemat(path+'NSR_groups.mat',{'groupId': group_id_NSR})
savemat(path+'RHD_groups.mat',{'groupId': group_id_RHD})

#%% Plot magnitude spectrum (filtered vs unfiltered ECG)
x=nk.signal_resample(ecg_NSR[10].reshape(-1), method="resample_method", sampling_rate=500, desired_sampling_rate=500)
x_mag, f = prep.fft_plot(Fs=500,N=len(X_NSR_filt[10]),samples=X_NSR_filt[10])
x, f2 = prep.fft_plot(Fs=500,N=len(X_NSR_filt[10]),samples=x)
df = pd.DataFrame({'[Original]': x, '[Filtered]':x_mag})
fig = px.line(df,x=f, y=df.columns[0:], color='variable').update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",title="ECG signal( FFT )",legend_title="Signal")
fig.show()


#%% Visualize PwRHD and Healthy controls(Normals)
# ##################################
# Plot PwRHD and Normals ECG Records
plt.tight_layout()
sns.set_style('white')
plt.figure(dpi=300)
plt.rcParams['figure.figsize'] = [15,6]
plt.rcParams['font.size'] = 20
desired_fs=500

fig, axs = plt.subplots(2, 2)
fig.suptitle('ECG signals from RHDECG dataset')
axs[0, 0].plot(np.arange(0,X_NSR_filt[9].size/desired_fs,1/desired_fs),X_NSR_filt[10], lw=1)
axs[0, 0].legend(['Normal ECG(Subject 5)'],loc="upper right",prop={'size': 14})
axs[1, 0].plot(np.arange(0,X_RHD_filt[14].size/desired_fs,1/desired_fs),X_RHD_filt[4], lw=1)
axs[1, 0].legend(['RHD ECG(Subject 12)'],loc="upper right",prop={'size': 14})
axs[0, 1].plot(np.arange(0,X_RHD_filt[49].size/desired_fs,1/desired_fs),X_RHD_filt[49], lw=1)
axs[0, 1].legend(['RHD ECG(Subject 100)'],loc="upper right",prop={'size': 14})
axs[1, 1].plot(np.arange(0,X_RHD_filt[136].size/desired_fs,1/desired_fs),X_RHD_filt[52], lw=1)
axs[1, 1].legend(['RHD ECG(Subject 120)'],loc="upper right",prop={'size': 14})
# Set common labels
fig.supxlabel('Time(sec)')
fig.supylabel('Amplitude (mV)')
fig.tight_layout()
plt.show()


#%% Load RWE features precomputed using the following matlab script
# '''
# matlab file path to compute relative wavelet energy features
# The relative wavelet energy(RWE) is cmputed from each ECG record using multiresolution analysis as shown in figure below. 
# use the following matlab snippet. DONOT FORGET to change the specific filepath and save directory for each class.
# #http://localhost:8888/notebooks/OneDrive%20-%20KU%20Leuven/Experiments/RHDECG/compute_RWE.m

# close all;
# clear all;
# %% Load dataset
# x_nsr=load('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\RHDECG\Data\NSR_X.mat'); 
# x_rhd=load('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\RHDECG\Data\RHD_X.mat');
# x_ptb=load('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\Untitled Folder\PTB_nsrUnder35.mat');

# % get the size
# s=size(x.ECGRecord);

# %% Compute relative wavelet energy  and save the output RWE
# [relative_energy_nsr,relative_Shanon_energy_nsr] = computeRWE(x_nsr);
# [relative_energy_rhd,relative_Shanon_energy_rhd] = computeRWE(x_rhd);
# [relative_energy_ptb,relative_Shanon_energy_ptb] = computeRWE(x_ptb);

# % SE = -|a[n]|log(|a[n]|) %-1*abs(energy_by_scales).*log2(abs(energy_by_scales));
# save('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\RHDECG\Data\relative_energy_NSR', 'relative_energy_nsr', '-v7.3')
# save('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\RHDECG\Data\relative_energy_RHD', 'relative_energy_rhd', '-v7.3')
# save('C:\Users\u0143922\OneDrive - KU Leuven\Experiments\RHDECG\Data\relative_energy_PTB', 'relative_energy_ptb', '-v7.3')

# function [relative_energy,relative_Shanon_energy] = computeRWE(x);
#     %% Compute relative wavelet energy
#     s=size(x);
#     energy=zeros(s(1),10); % we decompose the signal into 6 levels
#     relative_energy=zeros(s(1),10);
#     Shannon_energy=zeros(s(1),10);
#     relative_Shanon_energy=zeros(s(1),10);
#     for j=1:s(1)
#         Input_Signal=x(j,:);  % read 10sec record 
#         Input_Signal = Input_Signal-mean(Input_Signal);
#         %%%%%%%%%%%%%%%%%%%%%%%%
#         % Compue relative energy of a signal using MODWT
#         % https://nl.mathworks.com/help/wavelet/ref/modwt.html
#         level=9;
#         ecg_coef = modwt(Input_Signal,level,'db4',TimeAlign=true);
#         sig_len=length(ecg_coef);
#         energy_by_scales = 1/sig_len * sum(ecg_coef.^2,2);
#         Levels = {'D1';'D2';'D3';'D4';'D5';'D6';'D7';'D8';'D9';'A9'};
#         energy_total = sum(energy_by_scales);
#         relative_energy_by_scales = energy_by_scales./sum(energy_by_scales) ; % take percentages of relative energy
#         shannon_energy=-1*abs(energy_by_scales).*log2(abs(energy_by_scales));
#         relative_shannon_energy = shannon_energy./sum(shannon_energy); 
#         table(Levels,energy_by_scales,relative_energy_by_scales,relative_shannon_energy);
        
#         energy(j,:)=energy_by_scales;
#         relative_energy(j,:)=relative_energy_by_scales;
#         Shannon_energy(j,:) = shannon_energy;
#         relative_Shanon_energy(j,:) = relative_shannon_energy;
         
#     end

# '''

path=data_path
nsr_rel_energy = mat73.loadmat(path+'relative_energy_nsr_final.mat')
rhd_rel_energy = mat73.loadmat(path+'relative_energy_rhd_final.mat')
# ptb_rel_energy = mat73.loadmat('relative_energy_PTB_below35.mat')
# chf_rel_energy = mat73.loadmat('relative_energy_CHF.mat')

df_RWE_rhdecg = np.vstack((nsr_rel_energy['relative_energy'],rhd_rel_energy['relative_energy'])) 
# df_RWE_ptbrhdecg = np.vstack((nsr_rel_energy['relative_energy'],ptb_rel_energy['relative_Shanon_energy'],rhd_rel_energy['relative_energy'])) 
print('RWE features for both classes: ',df_RWE_rhdecg.shape)

df_RWE_rhdecg= pd.DataFrame(df_RWE_rhdecg[:,1:-1], columns=['D2','D3','D4','D5','D6','D7','D8','D9'])# the last scaling coef. were not considered
# df_RWE_ptbrhdecg= pd.DataFrame(df_RWE_ptbrhdecg[:,1:-1], columns=['D2','D3','D4','D5','D6','D7','D8','D9'])# the last scaling coef. were not considered
# df_RWE_chf= pd.DataFrame(chf_rel_energy['relative_energy'][:,1:-1], columns=['D2','D3','D4','D5','D6','D7','D8','D9'])# the last scaling coef. were not considered
df_RWE_rhdecg.head()


#%% Classification with PCA components using SVM

# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, roc_auc_score, roc_curve, auc, RocCurveDisplay 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = [5,5]

# Arrange Data (X, y)
ECG_RWE_X=np.array(df_RWE_rhdecg) #pca_comp
# ECG_RWE_X = np.array(pd.concat([df_RWE_rhdecg], axis=0, join='inner'))
# CHF_Y = np.array([2 for i in range(len(df_RWE_chf))])
ECG_RWE_Y=np.concatenate((np.zeros(len(nsr_rel_energy['relative_energy'])),np.ones(len(rhd_rel_energy['relative_energy']))),axis=0) #perSubject_avged

# # For CHF dataset
# groups_CHF=np.ones((len(CHF_Y),))
# rec_list_chf=[]
# for t in range(len(CHF_Y)):
# 	rec_list_chf.append([t+20001, 1])  
# rec_list_chf = np.array(rec_list_chf)

## Normalize [0,1] with minmax
scaler=MinMaxScaler()
Data_X_normalized = np.array([scaler.fit_transform(rec.reshape(-1,1)) for rec in ECG_RWE_X]) # Normalize [0,1]
print('X,Y shape:',(ECG_RWE_X.shape,ECG_RWE_Y.shape))

groups=np.hstack((group_id_NSR, group_id_RHD)).reshape(-1)
rec_list = np.vstack((rec_list_NSR, np.array(rec_list_RHD)))
print('groups: ',groups.shape)
print('Rec_list: ',rec_list.shape)




#HRV#################################################333
def compute_HRV_feat(ecg_signal):
    Fs=500
    HRV_feat=[]
    for i in range(len(ecg_signal)):    
        ecg_filt = nk.ecg_clean(ecg_signal[i], sampling_rate=Fs, method='neurokit')
        _, peaks = nk.ecg_peaks(ecg_filt, sampling_rate=Fs, show=True,method='pantompkins')
        hrv_time = nk.hrv_time(peaks, sampling_rate=Fs, show=False)
        HRV_feat.append(hrv_time)

    HRV_feat=pd.concat(HRV_feat)
    HRV_feat= HRV_feat.dropna(axis=1, how='all').reset_index(drop=True) # drop if there are NANs
    return HRV_feat

HRV_NSR = compute_HRV_feat(ecg_NSR[:,:5000]) # HRV features for Healthy controls
HRV_RHD = compute_HRV_feat(ecg_RHD[:,:5000]) # HRV features for RHD subjects
# cols=['D1','D2','D3','D4','D5','D6','A6','HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD','HRV_CVNN','HRV_CVSD','HRV_MedianNN','HRV_MadNN','HRV_MCVNN','HRV_IQRNN','HRV_Prc20NN','HRV_Prc80NN','HRV_pNN50','HRV_pNN20','HRV_MinNN','HRV_MaxNN','HRV_HTI','HRV_TINN']
HRV_NSR =HRV_NSR.drop(['HRV_MinNN','HRV_MaxNN','HRV_pNN20','HRV_MeanNN','HRV_SDNN','HRV_CVNN','HRV_MedianNN','HRV_MCVNN','HRV_IQRNN','HRV_CVSD','HRV_Prc80NN','HRV_HTI','HRV_TINN','HRV_MCVNN'],axis=1)
HRV_RHD =HRV_RHD.drop(['HRV_MinNN','HRV_MaxNN','HRV_pNN20','HRV_MeanNN','HRV_SDNN','HRV_CVNN','HRV_MedianNN','HRV_MCVNN','HRV_IQRNN','HRV_CVSD','HRV_Prc80NN','HRV_HTI','HRV_TINN','HRV_MCVNN'],axis=1)


# Now Merge the two features
RWE_HRV_NSR = np.hstack((nsr_rel_energy['relative_energy'][:,1:-1], HRV_NSR))
RWE_HRV_RHD = np.hstack((rhd_rel_energy['relative_energy'][:,1:-1], HRV_RHD))
RHDECG_HRV_X = np.vstack((HRV_NSR,HRV_RHD))

RHDECG_RWE_HRV_X = np.vstack((RWE_HRV_NSR,RWE_HRV_RHD))
print('RWE + HRV features(RHD) shape: ',RWE_HRV_NSR.shape)
print('RWE + HRV features(NSR) shape: ',RWE_HRV_RHD.shape)

## Normalize [0,1] with minmax
scaler=MinMaxScaler()
Data_X_normalized = np.array([scaler.fit_transform(rec.reshape(-1,1)) for rec in RHDECG_RWE_HRV_X]) 
RHDECG_RWE_HRV_X = np.squeeze(Data_X_normalized, axis=2)
RHDECG_RWE_Y=ECG_RWE_Y
print('Merged features (X,Y) shape:',(RHDECG_RWE_HRV_X.shape,RHDECG_RWE_Y.shape))


# '''
#%% Compute HRV related features

import preprocess_func as prep_hrv
from scipy.signal import find_peaks
from hrvanalysis.preprocessing import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis.extract_features import get_time_domain_features, get_frequency_domain_features, \
    get_poincare_plot_features,get_csi_cvi_features, get_sampen
#, plot_psd
plt.rcParams['figure.figsize']=[20,6]
# 1) template matching filter to pronounce R-peaks
# 2) R-peak detection
# 3) RR interval analysis
def compute_HRV_feat(ecgs):
    peaks_all=[]
    HRV_params_all=[]
    HRV_params=[]
    HRVs=[]
    timedomain_params=[]
    HRV_params_all=[]
    flat_params_list=[]
    fs=500
    for j in range(0,len(ecgs)): 
        # Filter the signal if not filtered
        ecg=ecgs[j,:] 
        _, peaks = nk.ecg_peaks(ecg, sampling_rate=500, correct_artifacts=True, method='neurokit')
        peaks=peaks['ECG_R_Peaks']#[1:-1]#+delay
        peaks_all.append(peaks)
        
        # Compute HRV features tiny
        #r_intervals_list = np.diff(peaks*1/fs)
        rr_intervals_list = np.diff(peaks)
        HRVs.append(prep_hrv.timedomain(rr_intervals_list))
        hrv = {}
        for d in HRV_params:
            hrv.update(d)   
        HRV_params.append(list(hrv.values()))
        
        
        #################################################################
        # HRV analysis module/paper
        #################################################################
        rpeaks_list=peaks#rr_intervals_list*1000 # convert to ms
        # This remove outliers from signal
        rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,  
                                                        low_rri=300, high_rri=2000, verbose=0) #30bpm - 150bpm
        
        interpolated_rr_intervals=rr_intervals_without_outliers
        # This remove ectopic beats from signal
        nn_intervals_list,ectopic = remove_ectopic_beats(rr_intervals=rr_intervals_without_outliers,method='acar',verbose=1)
        if ectopic >=3:
            nn_intervals_list=list(np.zeros((len(nn_intervals_list))))
            # print('Ectopic beats:  ',j)
    
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        
        time_domain_features = get_time_domain_features(interpolated_nn_intervals)
        timedomain_params.append([time_domain_features])
        flat_params_list = np.concatenate(timedomain_params).ravel()
        hrv_all = {}
        for d in flat_params_list:
            hrv_all.update(d) 
        HRV_params_all.append(list(hrv_all.values()))
    
    #cols=list(time_domain_features.keys())+list(frequency_domain_features.keys())+list(poincare_features.keys())+list(csi_cvi_features.keys())+list(samp_entropy.keys())
    cols=list(time_domain_features.keys())
    df_RR_feat = pd.DataFrame(HRV_params_all, columns=cols)
    df_RR_feat.replace([np.inf, -np.inf], 0, inplace=True)  # can be replace with np.nan    
    df_RR_feat.to_excel("df_HRV.xlsx",sheet_name='HRV_features_normalized')

    return df_RR_feat

# '''


# hrv_feats = pd.read_excel('df_HRV_amsalu_x.xlsx')
# HRV_NSR = hrv_feats.iloc[:138,:]
# HRV_RHD = hrv_feats.iloc[138:,:]
# RHDECG_RWE_Y=ECG_RWE_Y
# print(HRV_NSR.shape,HRV_RHD.shape)








#%% Compute Temporal features

# NSR_temporal=tempfeat.compute_temp(data_in=X_NSR_filt,samplingrate=500)
# RHD_temporal=tempfeat.compute_temp(data_in=X_RHD_filt,samplingrate=500)
temporal_feats = pd.read_excel(path+'rhdecg_temporal_final.xlsx')
# temporal_feats = temporal_feats.drop(['Diagnosis'], axis=1)
NSR_temporal = temporal_feats.iloc[:138,:]
RHD_temporal = temporal_feats.iloc[138:,:]
print(NSR_temporal.shape,RHD_temporal.shape)

features_temporal = np.vstack((np.array(NSR_temporal).reshape(len(NSR_temporal),-1),np.array(RHD_temporal).reshape(len(RHD_temporal),-1)))
print('Temporals: ',features_temporal.shape)
features_hrv = np.vstack((HRV_NSR,HRV_RHD))
print('HRV: ',features_hrv.shape)

RWE_HRV_temporal_NSR = np.hstack((nsr_rel_energy['relative_energy'][:,1:-1], HRV_NSR, np.array(NSR_temporal).reshape(len(NSR_temporal),-1)))
RWE_HRV_temporal_RHD = np.hstack((rhd_rel_energy['relative_energy'][:,1:-1],HRV_RHD, np.array(RHD_temporal).reshape(len(RHD_temporal),-1)))
RHDECG_RWE_temporal_X = np.vstack((RWE_HRV_temporal_NSR,RWE_HRV_temporal_RHD))

RHDECG_temporal_X = np.hstack((features_hrv,features_temporal))
RHDECG_RWE_temporal_X[np.isnan(RHDECG_RWE_temporal_X)] = 0
print('RWE + hrv + temporal features(RHD) shape: ',RWE_HRV_temporal_NSR.shape)
print('RWE + hrv + temporal features(NSR) shape: ',RWE_HRV_temporal_RHD.shape)

# ## Normalize [0,1] with minmax
# scaler=MinMaxScaler()
# Data_X_normalized = np.array([scaler.fit_transform(rec.reshape(-1,1)) for rec in RHDECG_RWE_temporal_X]) 
# RHDECG_RWE_temporal_X = np.squeeze(Data_X_normalized, axis=2)

# print('Merged features (X,Y) shape:',(RHDECG_RWE_temporal_X.shape,RHDECG_RWE_Y.shape))



#%% Define models to classify, fit and plot results
##################################################
# Model
########## (Table 5(a)-7) ##########

# !pip install xgboost
# load required classifer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedGroupKFold, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score,confusion_matrix, accuracy_score,classification_report, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay 
from sklearn.preprocessing import OneHotEncoder
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier
import shap

params_xgb = {
    'max_depth'  :[2,3,6],
    'min_child_weight' :[1,3,5],
    'learning_rate'  :[0.01,0.1,0.5,0.9],
    'gamma'  :[0,0.1,0.25,1],
    'reg_lambda'  :[0,1,10],
    'max_delta_step':[0.1,1,5],
    #'scale_pos_weight'  :[1,3,5],
    'subsample'     :[0.2,0.5,0.8]   
    }

def eval_ensemble(X, y, groups,cols):
    Data_X, Data_Y, groups = X, y, groups
    print('=========== Running Baseline Classifier ==============')
    
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    confusion_mat_rwehrv=0
    confusion_mat_rwe=0
    confusion_mat_hrv=0
    f1_fold_rwehrv=[]
    acc_fold_rwehrv=[]
    specificity_fold_rwehrv=[]
    sensitivity_fold_rwehrv=[]
    
    f1_fold_rwe=[]
    specificity_fold_rwe=[]
    sensitivity_fold_rwe=[]
    acc_fold_rwe=[] 
    
    f1_fold_hrv=[]
    specificity_fold_hrv=[]
    sensitivity_fold_hrv=[]
    acc_fold_hrv=[]    
    
    auc_fold_ens=[]
    fpr = dict()
    tpr = dict()
    roc_aucs = dict()
    mean_fpr = np.linspace(0, 1, 100)
    tprs_vis = []
    aucs_vis = []

    feat_importance1=[]
    feat_importance2=[]
    feat_importance3=[]

    
    fold_count=1
    splits=10
    folds = StratifiedGroupKFold(n_splits=splits, shuffle=False)
    estimator = XGBClassifier(n_estimators=100)
    for train_index, test_index in folds.split(Data_X, Data_Y, groups):
        X_train, X_test, y_train, y_test = Data_X[train_index], Data_X[test_index],Data_Y[train_index], Data_Y[test_index]
        rec_list_test=rec_list[test_index]
        classes_weights = class_weight.compute_sample_weight( class_weight='balanced',y=y_train)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, train_size=0.9, random_state=41)

        X_train=pd.DataFrame(X_train, columns=cols)
        X_val=pd.DataFrame(X_val, columns=cols)

        # Making the Classifer Ensemble with voting
        # Classifier1 = SVC(kernel='poly', probab1ility=True, C=1000, degree=5)
        # Classifier2 = AdaBoostClassifier(n_estimators=200, learning_rate=1)
        # Classifier = XGBClassifier(n_estimators=100,learning_rate=0.1, scale_pos_weight = .5)
        
        # Classifier = RandomForestClassifier(n_estimators= 100, max_depth=6, random_state=0)
        
        # Classifier5 = KNeighborsClassifier(n_neighbors=3, leaf_size=10)
        # Classifier6 = GaussianNB()
        # Classifier8 = RUSBoostClassifier(n_estimators=100, estimator=estimator)
        # Classifier = EasyEnsembleClassifier(random_state = 42, n_jobs = -1,n_estimators=300)
        # Classifier = GradientBoostingClassifier(n_estimators= 100, random_state = 0, subsample=0.6, max_features='sqrt')#,sampling_strategy = 0.5)
        # Classifier = BalancedRandomForestClassifier(n_estimators= 200, random_state = 0, n_jobs = -1, class_weight='balanced')

        # ensembleClassifier = VotingClassifier(
        #     estimators=[('SVM', Classifier1), ('ADB', Classifier2), ('XGB', Classifier3), ('RDF', Classifier4), ('KNN', Classifier5), ('GNB', Classifier6), ('RUSBoost', Classifier7), ('EasyEnsemble', Classifier8)],
        #     voting='soft')#, weights=[2, 5, 5, 5, 1, 1,1,1]) # voting='soft')

        # ensembleClassifier = VotingClassifier(
        #     estimators=[('XGB', Classifier3), ('EasyEnsemble', Classifier8)],
        #     voting='soft')
        
        # Classifier with gridsearch params in CV
        sample_weights = class_weight.compute_sample_weight(class_weight='balanced',y=Data_Y[train_index])
        ind, cnts = np.unique(sample_weights, return_counts=True)
        pos_weight =  cnts[0]/  cnts[1]
        Classifier = GridSearchCV(
            estimator=xgb.XGBClassifier(objective='binary:logistic',seed=41,early_stopping_rounds=20, colsample_bytree=0.75,scale_pos_weight=pos_weight),
            param_grid=params_xgb,
            scoring='f1',
            n_jobs=10,
            cv=3
            )              

        # HRV
        # ensembleClassifier.fit(X_train[:,9:], y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):

        # for clf, label in zip([Classifier3,Classifier8,ensembleClassifier], ['XGBoost', 'easyEnsemble','Ensemble']):
        #     clf.fit(X_train[:,9:], y_train)
        #     y_pred = clf.predict(X_test[:,9:])
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        
        
        Classifier.fit(X_train.iloc[:,8:13],y_train, eval_set=[(X_val.iloc[:,8:13],y_val)], verbose=False)
        # Classifier.fit(X_train.iloc[:,8:13], y_train) #hrv only
        y_pred = Classifier.predict(X_test[:,8:13]) #
        yt_ensembleclf,yp_ensembleclf,_,_ = prep.eval_binary_fold_classicalModel(Classifier, X_test[:,8:13], rec_list_test,y_test)
        report_ensemble=classification_report(yt_ensembleclf,yp_ensembleclf,output_dict=True, digits=3)

        # Collect the macro average scores
        f1_fold_hrv.append(report_ensemble['macro avg']['f1-score'])
        acc_fold_hrv.append(accuracy_score(yt_ensembleclf,yp_ensembleclf))
        tn, fp, fn, tp = confusion_matrix(yt_ensembleclf,yp_ensembleclf).ravel()
        sensitivity_fold_hrv.append((tp/(tp+fn)))
        specificity_fold_hrv.append((tn / (tn+fp)))
        cm=confusion_matrix(yt_ensembleclf,yp_ensembleclf)
        confusion_mat_hrv+=cm 
        print(f'F1-score fold {fold_count}: %0.3f\n' % np.round_(report_ensemble['macro avg']['f1-score'],decimals=3))
    
        # feature importance evalautaionHRV
        perm_importance_result_train = importance.permutation_importance(Classifier, X_train.iloc[:,8:13], y_train, n_repeats=10)
        feat_importance1.append(perm_importance_result_train)
                        
                        
        #RWE                
        # ensembleClassifier.fit(X_train[:,:9], y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):
        #     clf.fit(X_train[:,:9], y_train)
        #     y_pred = clf.predict(X_test[:,:9])
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        Classifier.fit(X_train.iloc[:,:8],y_train, eval_set=[(X_val.iloc[:,:8],y_val)], verbose=False)
        # Classifier.fit(X_train.iloc[:,:8], y_train) #RWE only
        y_pred = Classifier.predict(X_test[:,:8])
        yt_ensembleclf,yp_ensembleclf,_,_ = prep.eval_binary_fold_classicalModel(Classifier, X_test[:,:8], rec_list_test,y_test)
        report_ensemble=classification_report(yt_ensembleclf,yp_ensembleclf,output_dict=True, digits=3)

        # Collect the macro average scores
        f1_fold_rwe.append(report_ensemble['macro avg']['f1-score'])
        acc_fold_rwe.append(accuracy_score(yt_ensembleclf,yp_ensembleclf))
        tn, fp, fn, tp = confusion_matrix(yt_ensembleclf,yp_ensembleclf).ravel()
        sensitivity_fold_rwe.append((tp/(tp+fn)))
        specificity_fold_rwe.append((tn / (tn+fp)))
        cm=confusion_matrix(yt_ensembleclf,yp_ensembleclf)
        confusion_mat_rwe+=cm 
        print(f'F1-score fold {fold_count}: %0.3f\n' % np.round_(report_ensemble['macro avg']['f1-score'],decimals=3))
        # feature importance evalautaionRWE
        perm_importance_result_train = importance.permutation_importance(Classifier, X_train.iloc[:,:8], y_train, n_repeats=10)
        feat_importance2.append(perm_importance_result_train)
                        
        # RWE + HRV                
        # ensembleClassifier.fit(X_train, y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):
        #     clf.fit(X_train, y_train)
        #     y_pred = clf.predict(X_test)
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        Classifier.fit(X_train,y_train, eval_set=[(X_val,y_val)], verbose=False)
        # Classifier.fit(X_train, y_train) # RWE + HRV
        y_pred = Classifier.predict(X_test)
        yt_ensembleclf,yp_ensembleclf,_,_ = prep.eval_binary_fold_classicalModel(Classifier, X_test, rec_list_test,y_test)
        report_ensemble=classification_report(yt_ensembleclf,yp_ensembleclf,output_dict=True, digits=3)

        # Collect the macro average scores
        f1_fold_rwehrv.append(report_ensemble['macro avg']['f1-score'])
        acc_fold_rwehrv.append(accuracy_score(yt_ensembleclf,yp_ensembleclf))
        tn, fp, fn, tp = confusion_matrix(yt_ensembleclf,yp_ensembleclf).ravel()
        sensitivity_fold_rwehrv.append((tp/(tp+fn)))
        specificity_fold_rwehrv.append((tn / (tn+fp)))
        cm=confusion_matrix(yt_ensembleclf,yp_ensembleclf)
        confusion_mat_rwehrv+=cm 
        print(f'F1-score fold {fold_count}: %0.3f\n' % np.round_(report_ensemble['macro avg']['f1-score'],decimals=3))
        
                        
        viz = RocCurveDisplay.from_predictions(
                yt_ensembleclf, 
                yp_ensembleclf,
                name=f"ROC fold {fold_count}",
                alpha=0.6,
                lw=1,
                ax=ax,
                linestyle='--',
            )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs_vis.append(interp_tpr)
        aucs_vis.append(viz.roc_auc)
        print(classification_report_imbalanced(yt_ensembleclf,yp_ensembleclf, target_names = ["Normals", "PwRHD"], digits = 3))

        # feature importance evalautaion_RWEHRV
        perm_importance_result_train = importance.permutation_importance(Classifier, X_train, y_train, n_repeats=10)
        feat_importance3.append(perm_importance_result_train)
        
        
        feat_means_Fold=np.vstack((
            np.concatenate([item['importances_mean'] for item in feat_importance3]).ravel().reshape(-1,len(cols))
        ))
        #Std scores of features importance
        feat_std_Fold=np.vstack((
            np.concatenate([item['importances_std'] for item in feat_importance3]).ravel().reshape(-1,len(cols))
        ))
        df_featImportanceFold = pd.DataFrame(feat_means_Fold, columns=cols)
        df_featImportanceFold=df_featImportanceFold.clip(lower=0) #getrid of -ve importance values
        indices = np.argsort(np.mean(np.concatenate(feat_means_Fold).ravel().reshape(-1,len(cols)),axis=0))

        # df_featImportanceFold1
        # Plot Feature importances
        plt.rcParams['font.size'] = 30
        plt.rcParams['figure.figsize'] = [40,8]
        df=df_featImportanceFold.reindex(df_featImportanceFold.mean().sort_values( ascending=False).index, axis=1)
        df2=df.std()

        # print(df1.head())
        dff=df.iloc[:,:]
        plt.figure(dpi=400)
        plt.bar(np.arange(dff.shape[1]), dff.mean(),capsize=6, color='orange')
        plt.xticks(range(len(dff.columns)), dff.columns)
        plt.xticks(rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Feature Importance')
        plt.title('Features importance graph of RWE & temporal ECG features')
        my_colors = 'k'#, 'b', 'b', 'b', 'k', 'k', 'b', 'b', 'k', 'k', 'k','b']
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
            ticklabel.set_color(tickcolor)
        plt.show()
        clf_model=Classifier
        clf_model=xgb.XGBClassifier(gamma= Classifier.best_params_['gamma'], learning_rate= Classifier.best_params_['learning_rate'],
                                max_depth= Classifier.best_params_['max_depth'], min_child_weight= Classifier.best_params_['min_child_weight'],
                                reg_lambda= Classifier.best_params_['reg_lambda'], max_delta_step= Classifier.best_params_['max_delta_step'], 
                                subsample= Classifier.best_params_['subsample'], objective='binary:logistic').fit(X_train, y_train)
        
        explainer = shap.Explainer(clf_model, X_test)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        # shap.summary_plot(shap_values, val_x, plot_type="bar")

        p=pd.DataFrame(shap_values,columns=X_train.columns)
        print(fold_count,fold_count)
        if fold_count<=1:
            pp=p
        else:
            pp=pd.concat([pp,p])
        
        
        fold_count+=1

        
        
        
        
    mean_tpr = np.mean(tprs_vis, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_vis)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
        lw=3,
        alpha=0.9)

    std_tpr = np.std(tprs_vis, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="aquamarine",
        alpha=0.2,
        label=r"$\pm$1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean AUROC (Ensemble)",
    )
    plt.rcParams.update({'font.size': 14})
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
    plt.close(fig)
    print('Average scores of folds (HRV):')
    print(f'> Sensitivity: {np.round_(np.mean(sensitivity_fold_hrv),decimals=3)}(+- {np.round_(np.std(sensitivity_fold_hrv),decimals=3)})')
    print(f'> Specificity: {np.round_(np.mean(specificity_fold_hrv),decimals=3)}(+- {np.round_(np.std(specificity_fold_hrv),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold_hrv),decimals=3)} (+- {np.round_(np.std(f1_fold_hrv),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(acc_fold_hrv),decimals=3)} (+- {np.round_(np.std(acc_fold_hrv),decimals=3)})')


    print('Average scores of folds (RWE):')
    print(f'> Sensitivity: {np.round_(np.mean(sensitivity_fold_rwe),decimals=3)}(+- {np.round_(np.std(sensitivity_fold_rwe),decimals=3)})')
    print(f'> Specificity: {np.round_(np.mean(specificity_fold_rwe),decimals=3)}(+- {np.round_(np.std(specificity_fold_rwe),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold_rwe),decimals=3)} (+- {np.round_(np.std(f1_fold_rwe),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(acc_fold_rwe),decimals=3)} (+- {np.round_(np.std(acc_fold_rwe),decimals=3)})')


    print('Average scores of folds (RWETemporal):')
    print(f'> Sensitivity: {np.round_(np.mean(sensitivity_fold_rwehrv),decimals=3)}(+- {np.round_(np.std(sensitivity_fold_rwehrv),decimals=3)})')
    print(f'> Specificity: {np.round_(np.mean(specificity_fold_rwehrv),decimals=3)}(+- {np.round_(np.std(specificity_fold_rwehrv),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold_rwehrv),decimals=3)} (+- {np.round_(np.std(f1_fold_rwehrv),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(acc_fold_rwehrv),decimals=3)} (+- {np.round_(np.std(acc_fold_rwehrv),decimals=3)})')



    plt.figure(dpi=400)
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.figsize'] = [10,10]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_hrv, display_labels=['Normals','PwRHD'])
    disp.plot(cmap='Blues',colorbar=False)
    plt.title('Confusion matrix 10folds-CV \n(Temporal)')
    plt.grid(False)
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_rwe, display_labels=['Normals','PwRHD'])
    disp.plot(cmap='Blues',colorbar=False)
    plt.title('Confusion matrix 10folds-CV \n(RWE)')
    plt.grid(False)
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_rwehrv, display_labels=['Normals','PwRHD'])
    disp.plot(cmap='Blues',colorbar=False)
    plt.title('Confusion matrix 10folds-CV \n(Temporal+RWE)')
    plt.grid(False)
    plt.show()
    
    plt.rcParams['figure.figsize'] = [40,8]
    pp.loc['Feature_import_total']=pp[X_train.columns].sum()
    df=pd.DataFrame(pp.iloc[-1]*10e15,X_train.columns)
    scaler = MinMaxScaler()
    df[['Feature_import_total']] = scaler.fit_transform(df[['Feature_import_total']])
    df.sort_values(by='Feature_import_total', ascending=False).head(25).plot.bar()

    
    return feat_importance1,feat_importance2,feat_importance3   


cols=['RWE_D2','RWE_D3','RWE_D4','RWE_D5','RWE_D6','RWE_D7','RWE_D8','RWE_D9','HRV_RMSSD','HRV_SDSD','HRV_MadNN','HRV_Prc20NN','HRV_pNN50',
      'P_duration','QRS_duration','T_duration','PR','QT','PT','QTc','TpTe','iCEBc','TpTe_QTc'
]


#%%
# from sklearn import decomposition
# pca = decomposition.PCA(n_components=50)  # not for RWE features but we will see later
# X = pd.DataFrame(RHDECG_RWE_temporal_X, columns=cols)
# new_df=pca.fit_transform(X)
# print('PCA components: ',pca_comp.shape)
# cols_pca=pd.DataFrame(pca_comp.components_.T, index=X.columns)
# cols=cols_pca
# results1,results2,results3= eval_ensemble(new_df.values, RHDECG_RWE_Y, groups,cols)

########################
# feature selectionwith cross validation
#######################
# from sklearn.feature_selection import RFECV
# from sklearn.feature_selection import RFE
# clfrefe = RandomForestClassifier(random_state = 0) # Instantiate the algo
# # clfrefe =  GradientBoostingClassifier(random_state = 0, subsample=0.6, max_features='sqrt')#n_estimators= 100, random_state = 42, n_jobs = -1, sampling_strategy = 'auto', class_weight='balanced')
# # clfrefe = XGBClassifier()
# rfecv  = RFECV(estimator= clfrefe, step=1, cv=StratifiedKFold(3), scoring="f1_macro",min_features_to_select=20) # Instantiate the RFECV and its parameters
# X = pd.DataFrame(RHDECG_RWE_temporal_X, columns=cols)
# rfecv_select = rfecv.fit(RHDECG_RWE_temporal_X, RHDECG_RWE_Y)
# print("Optimal number of features : %d" % rfecv.n_features_)
# columns_to_remove = X.columns.values[np.logical_not(rfecv.support_)]
# new_df = X.drop(list(columns_to_remove), axis = 1)
# cols=new_df.columns.to_list()
# results1,results2,results3=eval_ensemble(new_df.values, RHDECG_RWE_Y, groups,cols)
#########################

# Train the model
results1,results2,results3=eval_ensemble(RHDECG_RWE_temporal_X, RHDECG_RWE_Y, groups,cols)

#%%


# min_features_to_select = 1
# rfc = RandomForestClassifier(random_state = 32)
# rfe = RFE(estimator=rfc, n_features_to_select= 50, step=1)

# fits= rfe.fit(X, RHDECG_RWE_Y)
# for i in range(RHDECG_RWE_temporal_X.shape[1]):
#     print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))




'''

results1,results2,results3=eval_ensemble(RHDECG_RWE_temporal_X[:,:], RHDECG_RWE_Y, groups,cols)
cols1=cols[:9]
cols2=cols[9:]
results=results3
# cols=cols1
#Mean scores of features importance
feat_means_Fold=np.vstack((
    np.concatenate([item['importances_mean'] for item in results]).ravel().reshape(-1,len(cols))
))
#Std scores of features importance
feat_std_Fold=np.vstack((
    np.concatenate([item['importances_std'] for item in results]).ravel().reshape(-1,len(cols))
))
df_featImportanceFold = pd.DataFrame(feat_means_Fold, columns=cols)
df_featImportanceFold=df_featImportanceFold.clip(lower=0) #getrid of -ve importance values
indices = np.argsort(np.mean(np.concatenate(feat_means_Fold).ravel().reshape(-1,len(cols)),axis=0))

# df_featImportanceFold1
# Plot Feature importances
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = [10,8]
df=df_featImportanceFold.reindex(df_featImportanceFold.mean().sort_values( ascending=False).index, axis=1)
df2=df.std()

# print(df1.head())
dff=df.iloc[:,:10]
plt.figure(dpi=800)
plt.bar(np.arange(dff.shape[1]), dff.mean(),capsize=6, color='orange')
plt.xticks(range(len(dff)), dff.columns)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('RWE & temporal related features importance graph')
my_colors = 'k'#, 'b', 'b', 'b', 'k', 'k', 'b', 'b', 'k', 'k', 'k','b']
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
    ticklabel.set_color(tickcolor)
plt.show()
'''


#%%
# Merged CNN Model with RWE features

from tensorflow import keras
from tensorflow.keras.layers import Multiply, Input, Conv1D, DepthwiseConv1D, SeparableConv1D, LeakyReLU, BatchNormalization, ReLU,  \
     Activation, Add, Dropout, MaxPool1D, GlobalAvgPool1D, AvgPool1D,Dense, Concatenate,Reshape,Flatten
from tensorflow.keras import Model
from keras.callbacks import History, ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

#  InceptionTime model

#https://github.com/hfawaz/InceptionTime
def _inception_module( input_tensor, stride=1, activation='linear', use_bottleneck=True,bottleneck_size=32, kernel_size=11):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []
    nb_filters = 32
    for i in range(len(kernel_size_s)):
        conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

def _shortcut_layer( input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def build_model( input_shape, num_classes, depth, use_bottleneck,bottleneck_size, kernel_size=11):
    use_residual = True
    input_layer = Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x, use_bottleneck= True,bottleneck_size=32)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAvgPool1D()(x)

    output_layer = Dense(num_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def model_inceptionTime(input_shape, num_classes,ks,nb_epochs):
    output_directory = './'
    nb_filters = 16
    use_residual = True
    use_bottleneck = True
    depth = 4
    kernel_size = ks - 1
    callbacks = None
    batch_size = 16
    bottleneck_size = 32
    build=True
    verbose = False
    num_classes=num_classes
    model = build_model(input_shape, num_classes,depth, use_bottleneck,bottleneck_size)
    return model


from tensorflow import keras
def plot_history_metrics(history: keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()


# InceptionTime reported (Table 5(b))
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, History
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    roc_curve, auc, RocCurveDisplay, accuracy_score
    
import model_defs as models

plt.figure(dpi=150)
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = [6,6]

########### Initialize variables ##############
BATCH_SIZE = 16
EPOCHS = 60

f1_fold_rhd2 = []
precision_fold_rhd2 = []
recall_fold_rhd2 = []
f1_fold_group = []
f1_fold_group = []
precision_fold_group = []
recall_fold_group = []
f1_fold_sl = []
precision_fold_sl = []
recall_fold_sl = []
acc=[]
acc_group=[]
aucs=[]
aucs_group=[]
cm_group = 0
cm_slice = 0 
cm_rhdonly=0
cm_slice=0
cm_all=0 
cm_subj_all=0
y_subj_all=0
fpr = dict()
tpr = dict()
roc_aucs = dict()

tprs_vis = []
aucs_vis = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(dpi=100)
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(30, 5))
fold_count=1
num_classes=2
n_splits = 10
print('\n\n ===================================================================================================')

########## Fit the model for each fold ######################

folds = StratifiedGroupKFold(n_splits=n_splits, shuffle= False)
fold_num=0
#for train_index, test_index in folds.split(Data_X,Data_Y):
groups=np.hstack((group_id_NSR, group_id_RHD)).reshape(-1)
rec_list = np.vstack((rec_list_NSR, np.array(rec_list_RHD)))
Data_X, Data_Y, groups = np.vstack((X_NSR_filt,X_RHD_filt)), RHDECG_RWE_Y, groups
for train_index, test_index in folds.split(Data_X, Data_Y, groups):
    X_train, X_test, y_train, y_test = Data_X[train_index], Data_X[test_index],Data_Y[train_index], Data_Y[test_index]

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                   classes=np.unique(np.array(Data_Y[train_index]).reshape(-1)),
                                                   y=np.array(Data_Y[train_index]).reshape(-1)) 
    

    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
    rec_list_test=rec_list[test_index]
   
    X_train = np.array([stats.zscore(rec.reshape(-1,1)) for rec in np.array(X_train)]) 
    X_test = np.array([stats.zscore(rec.reshape(-1,1)) for rec in np.array(X_test)]) 
    X_train = np.squeeze(X_train, axis=2)
    X_test = np.squeeze(X_test, axis=2)
    
    ##########  Split validation Data
    X, X_val, y, y_val = train_test_split(X_train, y_train.astype(int), test_size=0.1, random_state=127)
    ######## Add channel index => (Samples X Length X 1 )
    X = np.expand_dims(X,axis=2)
    X_val = np.expand_dims(X_val,axis=2)
    X_test = np.expand_dims(X_test,axis=2)

    print('Total train data: ', np.expand_dims(X_train,axis=2).shape)
    print('Total test data: ', X_test.shape)
    print('Total validation data: ', X_val.shape)
    print('Total train labels: ', y_train.shape )
    print('Total test labels: ', y_test.shape) 
    
    ################### Initialize the model and params ###############
    model_cnn = model_inceptionTime(input_shape=(X_train.shape[1],1), num_classes=2, ks=17, nb_epochs=EPOCHS)
    opt=tf.keras.optimizers.Adam()
    model_cnn.compile(optimizer=opt, loss=prep.categorical_focal_loss(), metrics=['accuracy',
                                                                                 keras.metrics.AUC(),
                                                                                  keras.metrics.Precision(), 
                                                                                  keras.metrics.Recall()])
    weight_dict = dict()
    for index,value in enumerate(class_weights):
        weight_dict[index] = value
    
    print(f"....... Running Fold {fold_count} .....\n" )
    save_dir='dir_to_save'
    save=True
    if save:
        saved = save_dir + "saved_clasifier_CNN.h5"
        hist = save_dir + 'training_history.csv'
        checkpointer = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=0, save_best_only=True)
        history = History()
    callbacks_list = [history, checkpointer, models.loss_history, models.lrate, models.stop]
    
    history = model_cnn.fit(X,
                            tf.cast(y, tf.float32),
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            shuffle=True,
                            validation_data=(X_val,tf.cast(y_val, tf.float32)),
                            class_weight = weight_dict,
                            callbacks = callbacks_list,
                            verbose=1)
    
    # Evaluate the model per fold
    model_cnn.load_weights(saved)
    yt_group,yp_group,_,_ = prep.eval_binary_fold(model_cnn,X_test, rec_list_test,y_test)
    print('Classification_report per subject:\n',classification_report(yt_group,yp_group))  

    report=classification_report(yt_group,yp_group,output_dict=True)  
    macro_precision_group =  np.round_(report['macro avg']['precision'] ,decimals=3)
    macro_recall_group = np.round_(report['macro avg']['recall'] ,decimals=3)   
    macro_f1_group = np.round_(report['macro avg']['f1-score'] ,decimals=3)
    f1_fold_group.append(macro_f1_group)
    precision_fold_group.append(macro_precision_group)
    recall_fold_group.append(macro_recall_group)
    acc_grp=accuracy_score(yt_group,yp_group)
    acc_group.append(acc_grp)
    cm_group += confusion_matrix(yt_group, yp_group)

    # Per-slice prediction
    model_cnn.load_weights(saved)
    pred = model_cnn.predict(X_test)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(tf.cast(y_test, tf.float32), axis=1)    
#     plt.rcParams['figure.figsize'] = [15,3]
#     for tt in range(len(y_pred)):
#         if y_pred[tt]!=y_true[tt]:
#             plt.title(f'Prediction [i,pred,true]: {[tt, y_pred[tt],y_true[tt]]}')
#             plt.plot(X_test[tt])
#             plt.show()
    # plot_history_metrics(history)
    
    viz = RocCurveDisplay.from_predictions(
            yt_group, 
            yp_group,
            name=f"ROC fold {fold_count}",
            alpha=0.6,
            lw=1,
            ax=ax,
            linestyle='--',
        )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_vis.append(interp_tpr)
    aucs_vis.append(viz.roc_auc)
    fold_count+=1

mean_tpr = np.mean(tprs_vis, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs_vis)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=3,
    alpha=0.9)

std_tpr = np.std(tprs_vis, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="aquamarine",
    alpha=0.2,
    label=r"$\pm$1 std. dev.",
)
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean AUROC (WaveletCNN)",
)
plt.rcParams.update({'font.size': 14})
ax.axis("square")
ax.legend(loc="lower right")
plt.show()
plt.close(fig)
print('Average scores of folds:')
print(f'> Precision: {np.round_(np.mean(precision_fold_group),decimals=3)}(+- {np.round_(np.std(precision_fold_group),decimals=3)})')
print(f'> Recall: {np.round_(np.mean(recall_fold_group),decimals=3)}(+- {np.round_(np.std(recall_fold_group),decimals=3)})')
print(f'> F1: {np.round_(np.mean(f1_fold_group),decimals=3)} (+- {np.round_(np.std(f1_fold_group),decimals=3)})')
print(f'> ROC_AUC: {np.round_(mean_auc,decimals=3)} (+- {np.round_(std_auc,decimals=3)})')
print(f'> Accuracy: {np.round_(np.mean(acc_group),decimals=3)} (+- {np.round_(np.std(acc_group),decimals=3)})')

#  Provide average scores
print('------------------------------------------------------------------------')
print('Score per fold (Subjects)')
for i in range(0, len(f1_fold_group)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - F1: {f1_fold_group[i]} - Precision: {precision_fold_group[i]} - Recall: {recall_fold_group[i]}')
print('------------------------------------------------------------------------')

plt.figure(dpi=600)
plt.rcParams.update({'font.size': 50})
plt.rcParams['figure.figsize'] = [10,10]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_group, display_labels=['Healthy','RHD'])
disp.plot(cmap='Blues',colorbar=False)
plt.title('Confusion matrix 10folds-CV \n(CNN: InceptionTime)')
plt.grid(False)
plt.show()


# Defining the Grad-CAM algorithm
def grad_cam(layer_name, data, model):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    last_conv_layer_output, preds = grad_model(data)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0))
    
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output * pooled_grads/2
    heatmap = tf.reduce_mean(heatmap, axis=(1))
    heatmap = np.expand_dims(heatmap,0)
    return heatmap
#%% Class activation map
# DB2 Class activation map from the input layer to the last Conv. layer
plt.figure(dpi=300)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(30,4))
layer_name = 'conv1d_255'
label = ['Normal', 'PwRHD']
cnt = 0
fs=500

X_t = np.array(X_test*1)
for i in X_t:
    data = np.expand_dims(i,0)#np.expand_dims(i.reshape(-1),0)
    pred_arry=model_cnn.predict(data)[0]
    y_p = np.argmax(pred_arry,axis=0)
    y_t = np.argmax(y_test[cnt],axis=0)

    if  y_p == y_t:
        # print([y_p, y_t])
        heatmap = grad_cam(layer_name,data, model_cnn)
        # print(f"Model prediction = {label[y_p]} ({np.round(pred_arry,3)}) , True label = {label[y_t]}")
        plt.figure(figsize=(30,4))
        plt.imshow(np.expand_dims(heatmap,axis=2),cmap='YlOrRd', aspect="auto", interpolation='nearest',extent=[0,5000/fs,i.min()/1000,i.max()/1000], alpha=0.3)
        plt.plot(np.arange(0,i.size/fs,1/fs),i/1000,'k')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude (mV)')
        plt.colorbar()
        plt.show()
    cnt +=1

