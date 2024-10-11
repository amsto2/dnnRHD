#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os, re, random
from itertools import cycle
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score, RocCurveDisplay 

from operator import itemgetter
# !pip install nested_dict
import nested_dict as nd
import neurokit2 as nk
import warnings
warnings.filterwarnings("ignore")

#plt.figure(dpi=150)

########### Data Purging
# The original data from the sensor is not cleaned and lots of artifacts in it. 
# After perfroming Bandpass filtering, we search for a clean segment of at least 30s length
def clean_data_normal(name, indx, temp_ecg):

    X_cleaned_normal = []

    clean_list_normal = nd.nested_dict()
    clean_list_normal['1001']['1'] = 0
    clean_list_normal['1003']['2'] = 6000
    clean_list_normal['1004']['3'] = 6000

    clean_list_normal['1005']['1'] = 15000
    clean_list_normal['1006']['8'] = 18000
    #clean_list_normal['1007']['1'] = 0    # the only record but noisy
    clean_list_normal['1009']['1'] = 10000
    clean_list_normal['1012']['2'] = 25000
    clean_list_normal['1013']['1'] = 1000

    clean_list_normal['1015']['2'] = 0

    clean_list_normal['1018']['1'] = 1000
    clean_list_normal['1020']['1'] = 10000
    clean_list_normal['1021']['7'] = 0
    clean_list_normal['1026']['2'] = 10000
    clean_list_normal['1028']['1'] = 15000
    clean_list_normal['1029']['1'] = 10000
    clean_list_normal['1031']['1'] = 0
    clean_list_normal['1035']['6'] = 20000
    clean_list_normal['1036']['2'] = 10000
    clean_list_normal['1037']['1'] = 0
    clean_list_normal['1038']['2'] = 3500
    clean_list_normal['1039']['1'] = 20000
    clean_list_normal['1042']['4'] = 0


    clean_list_normal['1004']['5'] = 0
    clean_list_normal['1013']['2'] = 10000
    clean_list_normal['1015']['2'] = 0
    clean_list_normal['1042']['3'] = 0

    

    begin_with=[1005,1006,1012,1036]
    inversed = [0]

    X_n=0
    clean_list = [0,]
    temp_ecg= temp_ecg[:-1,:]
    X_n = temp_ecg
        
    if indx[1] in inversed:
        X_n = -1*temp_ecg
    if str(indx[1]) in list(clean_list_normal.keys()) :
        len_begin = clean_list_normal[str(indx[1])][list(clean_list_normal[str(indx[1])].keys())[0]]
        file_idx = list(clean_list_normal[str(indx[1])].keys())[0]

        if file_idx == str(indx[4]):
            if indx[1] in begin_with:
                rec = X_n[:int(len_begin)]
            elif int(len_begin)==0:
                rec = X_n[:0]
            else:
                rec = X_n[int(len_begin):]
        else:
            rec = X_n
    else:
        rec = X_n
    X_cleaned_normal.append(rec)
    

    return X_cleaned_normal, rec


def clean_data_rhd(name, indx, temp_ecg):

    X_cleaned_rhd = []

    clean_list_rhd = nd.nested_dict()
    clean_list_rhd['3']['1'] = 6000
    clean_list_rhd['5']['2'] = 2000
    clean_list_rhd['10']['2'] = 0
    clean_list_rhd['102']['4'] = 0
    clean_list_rhd['104']['2'] = 0
    clean_list_rhd['104']['3'] = 0
    clean_list_rhd['110']['2'] = 0
    clean_list_rhd['114']['2'] = 0
    clean_list_rhd['118']['6'] = 0
    clean_list_rhd['18']['2'] = 0
    clean_list_rhd['44']['1'] = 1000   #The only record
    clean_list_rhd['50']['2'] = 0
    clean_list_rhd['55']['1'] = 4000
    clean_list_rhd['56']['2'] = 6000
    clean_list_rhd['57']['1'] = 8000
    clean_list_rhd['60']['2'] = 3000
    clean_list_rhd['62']['2'] = 7000
    clean_list_rhd['69']['3'] = 0
    clean_list_rhd['71']['7'] = 0
    clean_list_rhd['74']['4'] = 0
    clean_list_rhd['94']['2'] = 0


    clean_list_rhd['100']['1'] = 10000
    clean_list_rhd['112']['3'] = 20000
    clean_list_rhd['116']['1'] = 20000
    clean_list_rhd['117']['2'] = 8000
    clean_list_rhd['21']['1'] = 10000
    clean_list_rhd['33']['1'] = 30000
    clean_list_rhd['34']['1'] = 25000
    clean_list_rhd['40']['1'] = 20000
    clean_list_rhd['46']['1'] = 2000
    clean_list_rhd['51']['1'] = 20000
    clean_list_rhd['63']['1'] = 10000
    clean_list_rhd['64']['1'] = 15000 
    clean_list_rhd['65']['1'] = 10000
    clean_list_rhd['68']['1'] = 10000
    clean_list_rhd['70']['1'] = 20000
    clean_list_rhd['71']['3'] = 20000
    clean_list_rhd['72']['1'] = 30000
    clean_list_rhd['75']['1'] = 20000
    clean_list_rhd['77']['1'] = 20000
    clean_list_rhd['88']['1'] = 20000
    clean_list_rhd['84']['1'] = 10000
    clean_list_rhd['87']['1'] = 7000
    clean_list_rhd['90']['1'] = 30000
    clean_list_rhd['93']['1'] = 20000
    clean_list_rhd['95']['1'] = 16000
    clean_list_rhd['96']['2'] = 15000

    begin_with=[57,62,84,117,121]
    inversed = []

    X_n=0
    clean_list = [0,]

    temp_ecg= temp_ecg[:-1,:]
    X_n = temp_ecg
    if indx[1] in inversed:
        X_n = -1*temp_ecg
    if str(indx[1]) in list(clean_list_rhd.keys()) :
        len_begin = clean_list_rhd[str(indx[1])][list(clean_list_rhd[str(indx[1])].keys())[0]]
        file_idx = list(clean_list_rhd[str(indx[1])].keys())[0]
        if file_idx == str(indx[4]):
            if indx[1] in begin_with:
                rec = X_n[:int(len_begin)]
            elif int(len_begin)==0:
                rec = X_n[:0]
            else:
                rec = X_n[int(len_begin):]
        else:
            rec = X_n
    else:
        rec = X_n
    X_cleaned_rhd.append(rec)

    return X_cleaned_rhd, rec



# # Denoising and Normalization
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, sosfiltfilt, freqz,iirnotch, filtfilt,sosfreqz, savgol_filter
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from scipy import stats

def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y


def filter_sig_bandpass(data,low,high,fs,order):
        # Filter wander noise 0.05Hz is the wander noise and min-ECG freq is about 0.05Hz - bandpass filter
#         b, a = iirnotch(0.05, Q = 40, fs = fs)
#         filtered_ecg = filtfilt(b, a, data)
        
        filtered_ecg= butter_bandpass_filter(data=data, lowcut=low, highcut=high, fs=fs, order=order) # Lowpass cut-off=150Hz(adolescent/250Hz children) and highpass cut-off=0.05Hz/0.5Hz
        
        return filtered_ecg

def filter_sig_lowpass(data,lowcutoff,fs,order):
        # PLI noise at 50Hz   --- Bandtop filter
        # b, a = iirnotch(50, Q = 40, fs = fs)
        # filtered_ecg = filtfilt(b, a, data)
        # Filter wander noise 0.05Hz is the wander noise and min-ECG freq is about 0.05Hz - highpass filter
        sos = butter(N=order, Wn=lowcutoff, btype='low', output='sos', analog=False)
        filtered_ecg = sosfiltfilt(sos, data)
        
        return filtered_ecg

def filter_sig_highpass(data,highcutoff,fs,order):
        sos = butter(N=order, Wn=highcutoff, btype='high', output='sos', analog=False)
        filtered_ecg = sosfiltfilt(sos, data)
        return filtered_ecg
def baseline_correct(y, lam=10000000, p=0.1, niter=14):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y-z


def filter_normalize(data_in, peak_height, threshold):
    ############################## 1) FILTERING ################################################################
    #Apply Bandpass [0.05, 120]Hz
    #data_filt = filter_sig_bandpass(data=data_in,low=low, high=high,fs=sampling_rate, order=order)
    #data_filt = filter_sig_highpass(data=data_in,highcutoff=low, fs=sampling_rate, order=order)
    
    ###### NOOOOOOOOOO DC offset/ Baseline drift correction
    #filtered_signals = np.array(data_filt)
    filtered_signals = np.array(baseline_correct(data_in))
    #filtered_signals = savgol_filter(filtered_signals, 11, 3) # window size 15, polynomial order 5 

    ############################## 2) Finding R-peak  ###########################################################
    peaks_data,_ = signal.find_peaks(filtered_signals,distance=100,height=peak_height)

    ############################## 3) Find Outlier R-peak based on qualntiles  ##################################
    # boxplot
    df = pd.DataFrame({'Rpeaks': filtered_signals[peaks_data].T})
    # Find quantile ranges 
    quantiles = df.quantile([0.01, 0.25, 0.5, 0.75, threshold])

    ############################## 4) Remove outliers in interquantile range  ###################################
    upper_threshold=quantiles.iloc[4][0]
    cutoff_threshold=quantiles.iloc[1][0]
    lower_threshold=quantiles.iloc[0][0]
    mask_outliers = np.array(np.where(filtered_signals > cutoff_threshold) or np.where(filtered_signals < lower_threshold))
    outlier_peaks = [mask_outliers[0][ii] for ii in range(mask_outliers.shape[1]) if mask_outliers[0][ii] in peaks_data]
    if mask_outliers.shape[1] > 0:
        print('Number of R-peak Outliers : ',mask_outliers.shape[1])
        ############################## 5) Rescale after outlier removal  ###########################################
        range_max= cutoff_threshold #np.max(filtered_signals[~mask_outliers],axis=1)
        range_min=np.min(filtered_signals,axis=0)
        filtered_signals[mask_outliers]= cutoff_threshold
        X_std = (filtered_signals - range_min) / (range_max - range_min)
        X_scaled =  X_std * (range_max - range_min) + range_min
    else:
        X_std = (filtered_signals - np.min(filtered_signals)) / (np.max(filtered_signals) - np.min(filtered_signals))
        X_scaled =  X_std * (np.max(filtered_signals) - np.min(filtered_signals)) + np.min(filtered_signals)
    ############################## 6) Now Take (z-norm) for all datasamples  ######################################################
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X_scaled = scaler.fit_transform(X_scaled.reshape(-1, 1)).reshape(-1)
    #X_scaled = stats.zscore(filtered_signals)
    #X_scaled = (X_scaled-np.mean(X_scaled))/max(np.abs(X_scaled-np.mean(X_scaled)))  # (Rizal and Hadiyoso, 2015; Hadiyoso and Rizal, 2017)
    return X_scaled


########## Data Encoding
# Arrange an normalize data
def generate_data(X_train_filt,X_test_filt,y_train,y_test):
    #convert them to pandas
    Xtrain = pd.DataFrame(X_train_filt, columns = [np.arange(1,X_train_filt.shape[1]+1)])
    Xtest = pd.DataFrame(X_test_filt, columns = [np.arange(1,X_test_filt.shape[1]+1)])

    ecg_trainY_cleaned = pd.DataFrame(y_train, columns = ['Labels']).astype('int')
    ecg_testY_cleaned = pd.DataFrame(y_test, columns = ['Labels']).astype('int')

    # One-Hot encoding
    Ytrain = pd.get_dummies(ecg_trainY_cleaned.iloc[:,0]).to_numpy()
    Ytest = pd.get_dummies(ecg_testY_cleaned.iloc[:,0]).to_numpy()

    XtrainNorm = Xtrain
    XtestNorm = Xtest

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                      classes=np.unique(np.array(ecg_trainY_cleaned).reshape(-1)), 
                                                      y=np.array(ecg_trainY_cleaned).reshape(-1)) 
    print('class_weights: ', class_weights)

    return XtrainNorm, Ytrain, XtestNorm, Ytest, class_weights


### Plots and fold average results
def params_init():
    confusion_mat_all=0
    cm_slices=0
    acc_fold=[]
    aucs_fold=[]
    f1_fold=[]
    precision_fold=[]
    recall_fold=[]
    fpr = dict()
    tpr = dict()
    roc_aucs = dict()
    cmatrix=0
    
    tprs_vis = []
    aucs_vis = []
    fold_count = 1

def getPerformResults(yt_group,yp_group):
#     print('Classification_report per subject:\n',classification_report(yt_group,yp_group))
    report=classification_report(yt_group,yp_group,output_dict=True, digits=4) 
    # Evaluating the accuracy of the model using the sklearn functions
    f1_fold.append(report['macro avg']['f1-score'])
    precision_fold.append(report['macro avg']['precision'])
    recall_fold.append(report['macro avg']['recall'])
#     confusion_mat_all+=confusion_matrix(yt_group, yp_group)
    acc_fold.append(accuracy_score(yt_group, yp_group))

    enc = OneHotEncoder()
    y_t = enc.fit_transform(yt_group.reshape(-1,1)).toarray()
    y_p = enc.fit_transform(yp_group.reshape(-1,1)).toarray()
    
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_t[:,i], y_p[:, i])
        roc_aucs[i] = auc(fpr[i], tpr[i])   
    aucs_fold.append(roc_aucs[i])

#     # Confusion matrix for each slices
#     y_pred = Classifier.predict(X_test)
#     cm_sl=confusion_matrix(y_test,y_pred)
#     cm_slices+=cm_sl

#     return precision_fold, recall_fold, f1_fold, aucs_fold, acc_fold, confusion_mat_all, cm_slices

def showResults():
    print('------------------------------------------------------------------------')
    print('Average scores:')
    print(f'> Precision: {np.round_(np.mean(precision_fold),decimals=3)}(+- {np.round_(np.std(precision_fold),decimals=3)})')
    print(f'> Recall: {np.round_(np.mean(recall_fold),decimals=3)}(+- {np.round_(np.std(recall_fold),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold),decimals=3)} (+- {np.round_(np.std(f1_fold),decimals=3)})')
    print(f'> AUC: {np.round_(np.mean(aucs_fold),decimals=3)}(+- {np.round_(np.std(aucs_fold),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(np.array(acc_fold)),decimals=3)}(+- {np.round_(np.std(np.array(acc_fold)),decimals=3)})')
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm_slices, display_labels=['Normal','RHD'])
#     disp.plot()
#     plt.title('Confusion Matrix for all ECG recordings')
    plt.grid(False)
plt.figure(dpi=300)
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(20, 8))
fold_count=1
tprs_vis = []
aucs_vis = []

def compute_ROC(yt_group, yp_group):
    # Plot Mean ROC_for folds
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
#     fold=fold+1
    return tprs_vis, aucs_vis, mean_fpr

def plot_ROC():
#     plt.figure(dpi=300)
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
        alpha=0.9,
    )

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
        title=f"Mean ROC curve (Ensemble)",
    )
#     plt.rcParams.update({'font.size': 16})
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
    


#Post-processing for Subject-wise prediction
from operator import itemgetter
from itertools import cycle

def groupby_mean(a):
    # Sort array by groupby column
    b = a[a[:,0].argsort()]

    # Get interval indices for the sorted groupby col
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])

    # Get counts of each group and sum rows based on the groupings & hence averages
    counts = np.diff(idx)
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts.astype(float)[:,None]

    # Finally concatenate for the output in desired format
    return np.round_(np.c_[b[idx[:-1],0],avg].astype(int))

def eval_binary_fold_classicalModel(model, test_data, rec_list_ext, test_lbls):    
    
    pred = model.predict(test_data)    
    rec_list2=rec_list_ext
#     print([pred,test_lbls])
#     print(rec_list_ext)
    # for i in range(x_sort[0][0]):
    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]
#         print(c , c+rec_list2[j][1])
        pred_k = pred[c : c+rec_list2[j][1]]
        true_k = test_lbls[c : c+rec_list2[j][1]]
        #print('true_k', [j, true_k,pred_k])
        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)

        y_pred = np.argmax(np.array([normal_pred,rhd_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))
   
        track_list.append(rec_list2[j][0])
    
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg  #, y_t_ptb, y_p_ptb


def eval_ptb_fold_classicalModel(model, test_data, rec_list_ext, test_lbls):    
    
    pred = model.predict(test_data)    
    rec_list2=rec_list_ext
#     print([pred,test_lbls])
#     print(rec_list_ext)
    # for i in range(x_sort[0][0]):
    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]
#         print(c , c+rec_list2[j][1])
        pred_k = pred[c : c+rec_list2[j][1]]
        true_k = test_lbls[c : c+rec_list2[j][1]]
        #print('true_k', [j, true_k,pred_k])
        NSR_pred = np.count_nonzero(pred_k ==0)
        MI_pred =  np.count_nonzero(pred_k ==1)
        STTC_pred =  np.count_nonzero(pred_k ==2)
        CD_pred =  np.count_nonzero(pred_k ==3)
        HYP_pred =  np.count_nonzero(pred_k ==4)

        y_pred = np.argmax(np.array([NSR_pred,MI_pred,STTC_pred,CD_pred,HYP_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))
   
        track_list.append(rec_list2[j][0])
    
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg  #, y_t_ptb, y_p_ptb



#Plots
def my_ROC_plot(y_true,y_pred):

    # Plot linewidth.
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
      fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(2):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= 2
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = [8,5]
    plt.plot(fpr["micro"], tpr["micro"],
          label='micro ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
          color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
          label='macro ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(2), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
              label='ROC curve of class {0} (area = {1:0.2f})'
              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Performance: ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


#Post-processing for Subject-wise prediction
def groupby_mean(a):
    # Sort array by groupby column
    b = a[a[:,0].argsort()]

    # Get interval indices for the sorted groupby col
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])

    # Get counts of each group and sum rows based on the groupings & hence averages
    counts = np.diff(idx)
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts.astype(float)[:,None]

    # Finally concatenate for the output in desired format
    return np.round_(np.c_[b[idx[:-1],0],avg].astype(int))

def eval_model2_fold(model, test_data, rec_list_ext, test_lbls):    
    
    model_name = model
#     X_test = np.expand_dims(X_test,axis=2)
    pred = model_name.predict(test_data)    
    rec_list2=rec_list_ext
    # print(pred)
    # for i in range(x_sort[0][0]):
    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
    #     print('J= ',j)
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]
    #     print(c)
    #     print(c,c+rec_list[j][1])
        pred_k = np.argmax(pred[c : c+rec_list2[j][1]], axis=1)
        true_k = np.argmax(test_lbls[c : c+rec_list2[j][1]], axis=1)

#         print([true_k])
        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)
        #if verbose==1:
            #print(f' Subject: {rec_list2[j][0]}  Predicted as [Normal:{normal_pred}, RHD:{rhd_pred}]  (Y_true:{true_k}, Y_pred:{list(pred_k)})')
        y_pred = np.argmax(np.array([normal_pred,rhd_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))
        else:
            y_pred_ptb.append((rec_list2[j][0],y_pred))
            y_true_ptb.append((rec_list2[j][0],true_k[0]))    
        track_list.append(rec_list2[j][0])
#             print(rec_list2[j][1], rec_list_ext[j])
        # print([true_k], np.argmax(pred[c : c+rec_list2[j][1]], axis=1),(j))
        # if j in [1328,1329,1330,1331,1341,1342,1343,1344,1345]:
            # print('--------------', [rec_list2[j][0], pred[c : c+rec_list2[j][1]]])
            # print('\n')
    # print(y_pred_all)
    # print(y_true_all)    
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_ptb, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_ptb, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_ptb, y_p_ptb = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg, y_t_ptb, y_p_ptb


def eval_model3_fold(model, test_data, rec_list_ext, test_lbls):
    model_name = model
    pred = model_name.predict(test_data) 
    rec_list2= rec_list_ext
    y_pred_all = []
    y_true_all = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    c=0
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]

        pred_k = np.argmax(np.mean(pred[c : c+rec_list2[j][1]],axis=0), axis=0)
        true_k = np.argmax(test_lbls[c : c+rec_list2[j][1]], axis=1)
        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)

        y_pred = np.argmax(np.array([normal_pred, rhd_pred]),axis=0)
        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        y_pred_once.append((rec_list2[j][0],y_pred))
        y_true_once.append((rec_list2[j][0],true_k[0]))


    y2x=np.array(sorted(y_true_once, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_once, key=itemgetter(0)))

    y_pred_subject_x=groupby_subjects_mean(y1x)
    y_true_subject_x=groupby_subjects_mean(y2x)

    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x

def groupby_subjects_mean(a):
    # Sort array by groupby column
    b = a[a[:,0].argsort()]

    # Get interval indices for the sorted groupby col
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])

    # Get counts of each group and sum rows based on the groupings & hence averages
    counts = np.diff(idx)
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts.astype(float)[:,None]

    # Finally concatenate for the output in desired format
    return np.round_(np.c_[b[idx[:-1],0],avg].astype(int))
def eval_multi_topk_fold(model, test_data, rec_list_ext, test_lbls):    
    
    model_name = model
#     X_test = np.expand_dims(X_test,axis=2)
    pred = model_name.predict(test_data)    
    rec_list2=rec_list_ext

    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]
        pred_k = np.argmax(pred[c : c+rec_list2[j][1]], axis=1)
        true_k = np.argmax(test_lbls[c : c+rec_list2[j][1]], axis=1)
        
        # compute top-k predictions
        prediction_probabilities_k = tf.math.top_k(pred[c : c+rec_list2[j][1]], k=2)   #top_k value
        dict_class_entries_k = prediction_probabilities_k.indices.numpy()       
        y_top_k=[]

        for l in range(len(true_k)):
            if dict_class_entries_k[l][0]==true_k[l]:
                p_k=dict_class_entries_k[l][0]
            elif dict_class_entries_k[l][1]==true_k[l]:
                p_k=dict_class_entries_k[l][1]
            else:
                p_k=np.argmax(pred[c : c+rec_list2[j][1]], axis=1)[l]
            y_top_k.append(p_k)
        
        pred_k=np.array(y_top_k)

        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)
        STTC_pred =  np.count_nonzero(pred_k ==2)
        HYP_pred =  np.count_nonzero(pred_k ==3)
        CD_pred =  np.count_nonzero(pred_k ==4)
        # MI_pred =  np.count_nonzero(pred_k ==5)
        y_pred = np.argmax(np.array([normal_pred,rhd_pred,STTC_pred,HYP_pred,CD_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))

        track_list.append(rec_list2[j][0])
  
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg  #, y_t_ptb, y_p_ptb

def eval_multi_fold(model, test_data, rec_list_ext, test_lbls):    
    
    model_name = model
#     X_test = np.expand_dims(X_test,axis=2)
    pred = model_name.predict(test_data)    
    rec_list2=rec_list_ext
    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]

        pred_k = np.argmax(pred[c : c+rec_list2[j][1]], axis=1)
        true_k = np.argmax(test_lbls[c : c+rec_list2[j][1]], axis=1)

        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)
        STTC_pred =  np.count_nonzero(pred_k ==2)
        HYP_pred =  np.count_nonzero(pred_k ==3)
        CD_pred =  np.count_nonzero(pred_k ==4)
        MI_pred =  np.count_nonzero(pred_k ==5)
        y_pred = np.argmax(np.array([normal_pred,rhd_pred,STTC_pred,HYP_pred,CD_pred,MI_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))
   
        track_list.append(rec_list2[j][0])
 
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]


    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg  #, y_t_ptb, y_p_ptb



def eval_binary_fold(model, test_data, rec_list_ext, test_lbls):    
    
    model_name = model
    pred = model_name.predict(test_data)    
    rec_list2=rec_list_ext

    pred_all = []
    y_pred_all = []
    y_true_all = []
    y_pred_ptb = []
    y_true_ptb = []
    y_pred_rhdecg = []
    y_true_rhdecg = []
    track_list = []
    y_pred_once=[]
    y_true_once=[]
    for j in range(len(rec_list2)):
        if j==0:
            c=0
        else:
            c+=rec_list2[j-1][1]

        pred_k = np.argmax(pred[c : c+rec_list2[j][1]], axis=1)
        true_k = np.argmax(test_lbls[c : c+rec_list2[j][1]], axis=1)

        normal_pred = np.count_nonzero(pred_k ==0)
        rhd_pred =  np.count_nonzero(pred_k ==1)

        y_pred = np.argmax(np.array([normal_pred,rhd_pred]),axis=0)

        y_pred_all.append((rec_list2[j][0],y_pred))
        y_true_all.append((rec_list2[j][0],true_k[0]))

        if rec_list_ext[j][0] <10000:
            y_pred_rhdecg.append((rec_list2[j][0],y_pred))
            y_true_rhdecg.append((rec_list2[j][0],true_k[0]))
   
        track_list.append(rec_list2[j][0])
    
    y2x=np.array(sorted(y_true_all, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_all, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_x, y_p_x = y_true_subject_x[:,1], y_pred_subject_x[:,1]

    y2x=np.array(sorted(y_true_rhdecg, key=itemgetter(0)))
    y1x=np.array(sorted(y_pred_rhdecg, key=itemgetter(0)))
    y_pred_subject_x=groupby_mean(y1x)
    y_true_subject_x=groupby_mean(y2x)
    y_t_rhdecg, y_p_rhdecg = y_true_subject_x[:,1], y_pred_subject_x[:,1]
    
    return y_t_x, y_p_x, y_t_rhdecg, y_p_rhdecg  #, y_t_ptb, y_p_ptb

def fft_plot(Fs,N,signals):
    win = np.hamming(N)                                                       
    x = signals[0:N] * win                            # Take a slice and multiply by a window
    sp = np.fft.rfft(x)                               # Calculate real FFT
    s_mag = np.abs(sp) * 2 / np.sum(win)              # Scale the magnitude of FFT by window and factor of 2,
                                                      # because we are using half of FFT spectrum
    s_db = 20 * np.log10(s_mag / N)                   # Convert to dB
    freq = np.arange((N / 2) + 1) / (float(N) / Fs)   # Frequency axis
    freq[0] = freq[0]/2                               # Donot multiply the DC-component, which is at indx-0
    return s_db, freq




# Feature importance
#https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html

# 3. Feature importance by permutation
def get_score_after_permutation(model, X, y, curr_feat):
    """return the score of model when curr_feat is permuted"""

    X_permuted = X.copy()
    col_idx = list(X.columns).index(curr_feat)
    # permute one column
    X_permuted.iloc[:, col_idx] = np.random.permutation(
        X_permuted[curr_feat].values
    )

    permuted_score = model.score(X_permuted, y)
    return permuted_score


def get_feature_importance(model, X, y, curr_feat):
    """compare the score when curr_feat is permuted"""

    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

    # feature importance is the difference between the two scores
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance


def permutation_importance(model, X, y, n_repeats=10):
    """Calculate importance score for each feature."""

    importances = []
    for curr_feat in X.columns:
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(
                get_feature_importance(model, X, y, curr_feat)
            )

        importances.append(list_feature_importance)

    return {
        "importances_mean": np.mean(importances, axis=1),
        "importances_std": np.std(importances, axis=1),
        "importances": importances,
    }

def plot_feature_importances(perm_importance_result, feat_name):
    """bar plot the feature importance"""

    fig, ax = plt.subplots()

    indices = perm_importance_result["importances_mean"].argsort()
    plt.barh(
        range(len(indices)),
        perm_importance_result["importances_mean"][indices],
        xerr=perm_importance_result["importances_std"][indices],
    )

    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])
