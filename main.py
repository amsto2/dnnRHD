# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 02:37:06 2024

@author: Amsalu Tomas Chuma
"""

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
import utils, models
import rhdecgPre_process as prep
import compute_temporal_feat as tempfeat
from data_loader import load_data

# load dataset and split the records based on subjects
ecg_NSR, NSR_labels, rec_list_NSR,group_id_NSR = load_data('normal')
ecg_RHD, RHD_labels, rec_list_RHD,group_id_RHD = load_data('rhd')       
print('Normal_RHDECG classes: ',ecg_NSR.shape)
print('RHD_RHDECG classes: ',ecg_RHD.shape)


#filter the signals 
Fs=500
sec=10
ecg=[]
for i in range(len(ecg_RHD)):
    ecg_filt=nk.ecg_clean(ecg_RHD[i], sampling_rate=Fs, method='neurokit', lowcut=0.5, highcut=100, order=4)
    ecg.append(ecg_filt)
X_RHD_filt = np.concatenate(ecg).ravel().reshape(-1,Fs*sec)
ecg=[]
for i in range(len(ecg_NSR)):
    ecg_filt=nk.ecg_clean(ecg_NSR[i], sampling_rate=Fs, method='neurokit', lowcut=0.5, highcut=100)
    ecg.append(ecg_filt)
X_NSR_filt = np.concatenate(ecg).ravel().reshape(-1,Fs*sec)

### Save as Matlab files for extracting RWE features
savemat('NSR_X.mat',{'ECGRecord': X_NSR_filt})
savemat('RHD_X.mat',{'ECGRecord': X_RHD_filt})
savemat('NSR_groups.mat',{'groupId': group_id_NSR})
savemat('RHD_groups.mat',{'groupId': group_id_RHD})


#####################################
# Extract RWE features of both Normals and RHD classes using computeRWE.m file in matlab.
#####################################

#%% Load RWE features
nsr_rel_energy = mat73.loadmat('relative_energy_NSR.mat')
rhd_rel_energy = mat73.loadmat('relative_energy_RHD.mat')
# nsr_rel_energy_Shannon = mat73.loadmat(path+'relative_Shanon_energy_NSR.mat')
# rhd_rel_energy_Shannon = mat73.loadmat(path+'relative_Shanon_energy_RHD.mat')
# ptb_rel_energy = mat73.loadmat('relative_energy_PTB_below35.mat')
# chf_rel_energy = mat73.loadmat('relative_energy_CHF.mat')

# Convert it to dataframe
df_RWE_rhdecg = np.vstack((nsr_rel_energy['relative_energy'],rhd_rel_energy['relative_energy'])) 
print('RWE features for both classes: ',df_RWE_rhdecg.shape)

df_RWE_rhdecg= pd.DataFrame(df_RWE_rhdecg[:,1:-1], columns=['D1','D2','D3','D4','D5','D6','D7','D8','D9'])# the last scaling coef. were not considered
df_RWE_rhdecg.head()



# perfrom classification with PCA components using SVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, roc_auc_score, roc_curve, auc, RocCurveDisplay 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = [5,5]

# Arrange Data (X, y) and merge RWE features
# RHDECG_RWE_X=np.array(df_RWE_rhdecg) #pca_comp
ECG_RWE_X = np.array(pd.concat([df_RWE_rhdecg], axis=0, join='inner'))
# CHF_Y = np.array([2 for i in range(len(df_RWE_chf))])
RHDECG_RWE_Y=np.concatenate((np.zeros(len(nsr_rel_energy['relative_energy'])),np.ones(len(rhd_rel_energy['relative_energy']))),axis=0) #perSubject_avged

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


#################################################
# HRV Features
#################################################
import preprocess_func as prep_hrv
from scipy.signal import find_peaks
from hrvanalysis.preprocessing import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis.extract_features import get_time_domain_features, get_frequency_domain_features, \
    get_poincare_plot_features,get_csi_cvi_features, get_sampen

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
HRV_NSR =HRV_NSR.drop(['HRV_SDNN','HRV_CVNN','HRV_MadNN','HRV_MCVNN','HRV_IQRNN','HRV_pNN50','HRV_HTI','HRV_TINN'], axis=1)
HRV_RHD =HRV_RHD.drop(['HRV_SDNN','HRV_CVNN','HRV_MadNN','HRV_MCVNN','HRV_IQRNN','HRV_pNN50','HRV_HTI','HRV_TINN'], axis=1)


# Now Merge the two features
RWE_HRV_NSR = np.hstack((nsr_rel_energy['relative_energy'][:,1:-1], HRV_NSR))
RWE_HRV_RHD = np.hstack((rhd_rel_energy['relative_energy'][:,1:-1], HRV_RHD))
RHDECG_HRV_X = np.vstack((HRV_NSR,HRV_RHD))

RHDECG_RWE_HRV_X = np.vstack((RWE_HRV_NSR,RWE_HRV_RHD))
print('RWE + HRV features(RHD) shape: ',RWE_HRV_NSR.shape)
print('RWE + HRV features(NSR) shape: ',RWE_HRV_RHD.shape)

## Normalize [0,1] with minmax
# scaler=MinMaxScaler()
# Data_X_normalized = np.array([scaler.fit_transform(rec.reshape(-1,1)) for rec in RHDECG_RWE_HRV_X]) 
# RHDECG_RWE_HRV_X = np.squeeze(Data_X_normalized, axis=2)
# print('Merged features (X,Y) shape:',(RHDECG_RWE_HRV_X.shape,RHDECG_RWE_Y.shape))




################################## Temporal

temporal_feats = pd.read_excel(path+'temp-feat-file')
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







##################################################3
# Model ( classical ML)
########## RWE features on RHDECG dataset (Table 5(a)) ##########

# !pip install xgboost
# load required classifer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score,confusion_matrix, accuracy_score,classification_report, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay 
from sklearn.preprocessing import OneHotEncoder
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier

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
    params_svm = {
        'C'  :[3,50,100],
        'kernel' :['rbf','poly'],
        #'gamma'  :[0.0001,0.001,0.01,0.1,1],
        'gamma': ['scale', 'auto'],
    }

    
    fold_count=1
    splits=10
    folds = StratifiedGroupKFold(n_splits=splits, shuffle=False)
    estimator = XGBClassifier(n_estimators=100)
    for train_index, test_index in folds.split(Data_X, Data_Y, groups):
        X_train, X_test, y_train, y_test = Data_X[train_index], Data_X[test_index],Data_Y[train_index], Data_Y[test_index]
        rec_list_test=rec_list[test_index]
        classes_weights = class_weight.compute_sample_weight( class_weight='balanced',y=y_train)

        # Initialize classifiers and making an Ensemble
        # Classifier1 = SVC(kernel='poly', probab1ility=True, C=1000, degree=5)
        # Classifier2 = AdaBoostClassifier(n_estimators=200, learning_rate=1)
        # Classifier = XGBClassifier(n_estimators=100,learning_rate=0.1, scale_pos_weight = .5)
        Classifier = RandomForestClassifier(n_estimators= 100, max_depth=6, random_state=0)
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

        sample_weights = class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
        ind, cnts = np.unique(sample_weights, return_counts=True)
        c_weights = {0: cnts[0], 1: cnts[1]}  # Class 0: , Class 1: 
        pos_weight = sum(c_weights.values()) / sum(c_weights.keys())
        Classifier = GridSearchCV(
            estimator=xgb.XGBClassifier(objective='binary:logistic',seed=41,early_stopping_rounds=20, colsample_bytree=0.75,scale_pos_weight=pos_weight),
            param_grid=params_xgb,
            scoring='f1',
            n_jobs=10,
            cv=3
            )    
        
        X_train=pd.DataFrame(X_train, columns=cols)
        ######################################
        # HRV features evaluation
        ######################################
        # ensembleClassifier.fit(X_train[:,9:], y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):

        # for clf, label in zip([Classifier3,Classifier8,ensembleClassifier], ['XGBoost', 'easyEnsemble','Ensemble']):
        #     clf.fit(X_train[:,9:], y_train)
        #     y_pred = clf.predict(X_test[:,9:])
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        Classifier.fit(X_train.iloc[:,9:], y_train) #rwe+hrv=8+16=24
        y_pred = Classifier.predict(X_test[:,9:]) #
        yt_ensembleclf,yp_ensembleclf,_,_ = prep.eval_binary_fold_classicalModel(Classifier, X_test[:,9:], rec_list_test,y_test)
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
        perm_importance_result_train = permutation_importance(Classifier, X_train.iloc[:,9:], y_train, n_repeats=10)
        feat_importance1.append(perm_importance_result_train)
                        
        ######################################                
        #RWE   
        ######################################
        # ensembleClassifier.fit(X_train[:,:9], y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):
        #     clf.fit(X_train[:,:9], y_train)
        #     y_pred = clf.predict(X_test[:,:9])
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        Classifier.fit(X_train.iloc[:,:9], y_train)
        y_pred = Classifier.predict(X_test[:,:9])
        yt_ensembleclf,yp_ensembleclf,_,_ = prep.eval_binary_fold_classicalModel(Classifier, X_test[:,:9], rec_list_test,y_test)
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
        perm_importance_result_train = permutation_importance(Classifier, X_train.iloc[:,:9], y_train, n_repeats=10)
        feat_importance2.append(perm_importance_result_train)

        ######################################
        # RWE + HRV                
        ######################################
        # ensembleClassifier.fit(X_train, y_train)
        # for clf, label in zip([Classifier1, Classifier2, Classifier3, Classifier4, Classifier5, Classifier6, Classifier7,Classifier8,ensembleClassifier], ['SVM', 'AdaBoost', 'XGBoost', 'RandForest', 'KNN', 'Gaussian Naive Bayes', 'RUSBoost','easyEnsemble','Ensemble']):
        #     clf.fit(X_train, y_train)
        #     y_pred = clf.predict(X_test)
        #     scores = accuracy_score(y_test,y_pred)
        #     print(f"Accuracy of {label}: %0.2f " % (scores))
        Classifier.fit(X_train, y_train)
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
        fold_count+=1
        print(classification_report_imbalanced(yt_ensembleclf,yp_ensembleclf, target_names = ["Normals", "PwRHD"], digits = 3))

        # feature importance evalautaion_RWEHRV
        perm_importance_result_train = permutation_importance(Classifier, X_train, y_train, n_repeats=10)
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
        dff=df.iloc[:,:20]
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
#     print('Average scores of folds (Ensemble RWEHRV):')
    print(f'> Sensitivity: {np.round_(np.mean(sensitivity_fold_hrv),decimals=3)}(+- {np.round_(np.std(sensitivity_fold_hrv),decimals=3)})')
    print(f'> Specificity: {np.round_(np.mean(specificity_fold_hrv),decimals=3)}(+- {np.round_(np.std(specificity_fold_hrv),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold_hrv),decimals=3)} (+- {np.round_(np.std(f1_fold_hrv),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(acc_fold_hrv),decimals=3)} (+- {np.round_(np.std(acc_fold_hrv),decimals=3)})')


#     print('Average scores of folds (Ensemble RWEHRV):')
    print(f'> Sensitivity: {np.round_(np.mean(sensitivity_fold_rwe),decimals=3)}(+- {np.round_(np.std(sensitivity_fold_rwe),decimals=3)})')
    print(f'> Specificity: {np.round_(np.mean(specificity_fold_rwe),decimals=3)}(+- {np.round_(np.std(specificity_fold_rwe),decimals=3)})')
    print(f'> F1: {np.round_(np.mean(f1_fold_rwe),decimals=3)} (+- {np.round_(np.std(f1_fold_rwe),decimals=3)})')
    print(f'> Accuracy: {np.round_(np.mean(acc_fold_rwe),decimals=3)} (+- {np.round_(np.std(acc_fold_rwe),decimals=3)})')


    print('Average scores of folds (Ensemble RWEHRV):')
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
    
    return feat_importance1,feat_importance2,feat_importance3   


cols=['RWE(60.3–100Hz)','RWE(30.2-64.6Hz)','RWE(15.1–32.3Hz)','RWE(7.5-16.2Hz)','RWE(3.7-8.08Hz)','RWE(1.8–4.04Hz)','RWE(0.9–2.02Hz)','RWE(0.5–1.01Hz)','SDSD','prcRR20','RMSSD','MADRR','CVSD',
      'QRS_duration(mean)','QRS_duration(median)',
      'T_duration(mean)','T_duration(median)',
      'RR(mean)','RR(std)','RR(Q2)','RR(Q50)','RR(Q98)','RR(range)','RR(median)',
      'TpTe(mean)','TpTe(median)',
      'iCEB(mean)','iCEB(median)',
      'iCEBc(mean)','iCEBc(median)',
]

######################################
# Feature selection 
######################################
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
clfrefe = RandomForestClassifier(random_state = 0) # Instantiate the algo
# clfrefe =  GradientBoostingClassifier(random_state = 0, subsample=0.6, max_features='sqrt')#n_estimators= 100, random_state = 42, n_jobs = -1, sampling_strategy = 'auto', class_weight='balanced')
# clfrefe = XGBClassifier()
rfecv  = RFECV(estimator= clfrefe, step=1, cv=StratifiedKFold(5), scoring="f1_macro",min_features_to_select=20) # Instantiate the RFECV and its parameters
X = pd.DataFrame(RHDECG_RWE_temporal_X, columns=cols)
rfecv_select = rfecv.fit(RHDECG_RWE_temporal_X, RHDECG_RWE_Y)
print("Optimal number of features : %d" % rfecv.n_features_)
columns_to_remove = X.columns.values[np.logical_not(rfecv.support_)]
new_df = X.drop(list(columns_to_remove), axis = 1)
cols=new_df.columns.to_list()
results1,results2,results3=eval_ensemble(new_df.values, RHDECG_RWE_Y, groups,cols)

# min_features_to_select = 1
# rfc = RandomForestClassifier(random_state = 32)
# rfe = RFE(estimator=rfc, n_features_to_select= 50, step=1)

# fits= rfe.fit(X, RHDECG_RWE_Y)
# for i in range(RHDECG_RWE_temporal_X.shape[1]):
#     print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))



'''
######################################
# Plot features importance 
######################################
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
my_colors = 'k']
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
    ticklabel.set_color(tickcolor)
plt.show()
'''



'''
# Merged CNN Model with RWE features

from tensorflow import keras
from tensorflow.keras.layers import Multiply, Input, Conv1D, DepthwiseConv1D, SeparableConv1D, LeakyReLU, BatchNormalization, ReLU,  \
     Activation, Add, Dropout, MaxPool1D, GlobalAvgPool1D, AvgPool1D,Dense, Concatenate,Reshape,Flatten
from tensorflow.keras import Model
from keras.callbacks import History, ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model


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
Data_X, Data_Y, groups = np.vstack((nsr_filt,rhd_filt)), RHDECG_RWE_Y, groups
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
    model_cnn = model_inceptionTime(input_shape=(X_train.shape[1],1), num_classes=2, ks=11, nb_epochs=EPOCHS)
    opt=tf.keras.optimizers.Adam()
    model_cnn.compile(optimizer=opt, loss=prep.categorical_focal_loss(), metrics=['accuracy',
                                                                                 keras.metrics.AUC(),
                                                                                  keras.metrics.Precision(), 
                                                                                  keras.metrics.Recall()])
    weight_dict = dict()
    for index,value in enumerate(class_weights):
        weight_dict[index] = value
    
    print(f"....... Running Fold {fold_count} .....\n" )
    save_dir=path
    save=True
    if save:
        saved = save_dir + "classifier_name.h5"
        hist = save_dir + 'classifier_name_training_history_cnn.csv'
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

    plot_history_metrics(history)
    
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

#########################################################
# Defining the Grad-CAM algorithm for model visualization
#########################################################
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

# Class activation map from the input layer to the last Conv. layer
plt.figure(dpi=600)
plt.rcParams.update({'font.size': 20})
layer_name = 'conv1d_219'
label = ['Normal', 'PwRHD']
cnt = 0
fs=500
X_t = np.array(X_test*100)
for i in X_t:
    data = np.expand_dims(i,0)#np.expand_dims(i.reshape(-1),0)
    pred_arry=model_cnn.predict(data)[0]
    y_p = np.argmax(pred_arry,axis=0)
    y_t = np.argmax(y_test[cnt],axis=0)

    if  y_p == y_t:
        print([y_p, y_t])
        heatmap = grad_cam(layer_name,data, model_cnn)
        print(f"Model prediction = {label[y_p]} ({np.round(pred_arry,3)}) , True label = {label[y_t]}")
        plt.figure(dpi=600)
        plt.figure(figsize=(30,4))
        plt.imshow(np.expand_dims(heatmap,axis=2),cmap='YlOrRd', aspect="auto", interpolation='nearest',extent=[0,5000/fs,i.min()/1000,i.max()/1000], alpha=0.5)
        plt.plot(np.arange(0,i.size/fs,1/fs),i/1000,'k')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude (mV)')
        plt.colorbar()
        plt.show()
    cnt +=1
'''
