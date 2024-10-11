import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Extract peak locations using neurokit
def peaks_finder(ecg_signal, Fs):
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=Fs)
    # Extract other-peaks locations
    _, waves_peak = nk.ecg_delineate(ecg_signal, 
                                     rpeaks, 
                                     sampling_rate=Fs, 
                                     method="dwt", 
                                     show=False)
    p_peaks_amp = ecg_signal[replace_nans(np.array(waves_peak['ECG_P_Peaks'])).astype(int)]
    t_peaks = replace_nans(np.array(waves_peak['ECG_T_Peaks'])).astype(int)
    t_peaks_amp = ecg_signal[replace_nans(np.array(waves_peak['ECG_T_Peaks'])).astype(int)]
    r_peaks = ecg_signal[replace_nans(rpeaks['ECG_R_Peaks'])]
    p_onsets = replace_nans(np.array(waves_peak['ECG_P_Onsets']))    
    p_offsets = replace_nans(np.array(waves_peak['ECG_P_Offsets']))
    r_onsets = replace_nans(np.array(waves_peak['ECG_R_Onsets']))
    r_peak = replace_nans(rpeaks['ECG_R_Peaks'])
    r_offsets = replace_nans(np.array(waves_peak['ECG_R_Offsets']))
    t_onsets = replace_nans(np.array(waves_peak['ECG_T_Onsets']))
    t_offsets = replace_nans(np.array(waves_peak['ECG_T_Offsets']))
    
    P_duration = np.subtract(p_offsets, p_onsets)
    QRS_duration = np.subtract(r_offsets, r_onsets)
    T_duration = np.subtract(t_offsets, t_onsets)

    PR = np.subtract(r_onsets, p_onsets)
#     print('PR: ',[PR])
    QT = np.subtract(t_offsets, r_onsets)
#     print('QT: ',[QT[:-1].astype(int)])
    ST = np.subtract(t_onsets, r_offsets)
    PT = np.subtract(t_offsets, p_onsets)
    STT = np.subtract(t_offsets, r_offsets)

#     print('ST: ',list(ST))
    RP_peak = np.subtract(r_peak, p_onsets)
    RR = np.diff(rpeaks['ECG_R_Peaks'])
#     print('RR: ',RR)
    QTc = (QT[:-1]/Fs) / np.sqrt((RR/Fs))  #to consider the len(RR) is N-1 len(QT) should be QT[:-1]
#     print('QTc: ',[(QTc*1000).astype(int)])
    TpTe = np.subtract(t_offsets,t_peaks)
#     print('TpTe: ',[t_peaks,t_offsets])
    iCEB = QT[:-1] / np.sqrt(QRS_duration[:-1])   
    
           
    return [p_peaks_amp, r_peaks, t_peaks_amp, P_duration, QRS_duration, T_duration, PR, QT, ST, PT, 
            STT, RP_peak, RR, QTc, TpTe, iCEB ]

def replace_nans(data):
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data



def parameters(data_in,p_peaks_all,r_peaks_all,t_peaks_all,P_duration_all,QRS_duration_all,T_duration_all,PR_all,QT_all,ST_all,PT_all,STT_all,RP_peak_all,RR_all,QTc_all,TpTe_all,iCEB_all):
    #  p_peaks_all,r_peaks_all,t_peaks_all,P_duration_all,QRS_duration_all,T_duration_all,PR_all,QT_all,ST_all,PT_all,STT_all,RP_peak_all,RR_all,QTc_all,TpTe_all,iCEB_all,RR_mean_all
    Fs=250
    ecg_signal = data_in
    # ptb length = ptb_250-1 because 25,47,146,1169th records are noisy
    for k in range(len(ecg_signal)):
        data_x=p_peaks_all[k]
    #     print(f'Subject_ID: {rec_list2[k][0]}')
    #print('\n= = = = = = = = = = = = = = = = =\n')
    for k in range(len(ecg_signal)):
        data_x=r_peaks_all[k]
        data_x=data_x[data_x > 0][1:]
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=t_peaks_all[k]
        data_x=data_x[data_x > 0][1:]
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=P_duration_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=QRS_duration_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
        #print(list(np.around(data_x,3)))
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=T_duration_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=PR_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=QT_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=ST_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')
        
    for k in range(len(ecg_signal)):
        data_x=PT_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=STT_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=RP_peak_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=RR_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')
        
    for k in range(len(ecg_signal)):
        data_x=QTc_all[k]
        data_x=data_x[data_x > 0][1:]
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=TpTe_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=iCEB_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')

    for k in range(len(ecg_signal)):
        data_x=TpTe_all[k][:-1]/QTc_all[k]
        data_x=data_x[data_x > 0][1:]
        data_x=data_x/Fs
    #print('\n= = = = = = = = = = = = = = = = =\n')


    feat_all=[]
    for k in range(len(T_duration_all)):
        Fs=250
        x=p_peaks_all[k]
        absx = [-v if v <0 else v for v in x ] 
    #     print(absx)
        x=x[x > 0][1:]
        df = pd.DataFrame(absx)
        feat1 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max    

        x=r_peaks_all[k]
        x=x[x > 0][1:]
        df = pd.DataFrame(x)
        feat2 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=t_peaks_all[k]
        absx = [-v if v <0 else v for v in x ] 
        x=x[x > 0][1:]
        df = pd.DataFrame(absx)
        feat3 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=P_duration_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat4 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=QRS_duration_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat5 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=T_duration_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat6 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=PR_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat7 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=QT_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat8 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=ST_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat9 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=PT_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat10 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=STT_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat11 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=RR_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat12 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=QTc_all[k]
        x=x[x > 0][1:]
        df = pd.DataFrame(x)
        feat13 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=TpTe_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat14 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=iCEB_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat15 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        x=TpTe_all[k][:-1]/QTc_all[k]
        x=TpTe_all[k][:-1]/QTc_all[k]
        x=x[x > 0][1:]
        x=x/Fs
        df = pd.DataFrame(x)
        feat16 = np.array([df.describe().iloc[1], df.describe().iloc[3], df.describe().iloc[7]]).reshape(-1) #mean. min, max

        feat_all.append([feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9,feat10,feat11,feat12,feat13,feat14,feat15,feat16])
    print(np.array(feat_all).shape)
    return feat_all

def compute_temp(data_in):
    p_peaks_all = []
    r_peaks_all = []
    t_peaks_all = []
    PR_all = []
    QT_all = []
    ST_all = []
    RR_all = []
    p_peaks = []
    r_peaks = []
    t_peaks = []
    P_duration_all = []
    QRS_duration_all = []
    T_duration_all = []
    PR_all = []
    QT_all = []
    ST_all = []
    PT_all = []
    STT_all = []
    RP_peak_all = []
    RR_all = []
    T_T_all = []
    QTc_all = []
    TpTe_all = []
    iCEB_all = []


    ecg_signal = data_in 

    for i in range(len(ecg_signal)):

        features = peaks_finder(ecg_signal[i], Fs=250)
        
        p_peaks_all.append(features[0])
        r_peaks_all.append(features[1])
        t_peaks_all.append(features[2])
        P_duration_all.append(features[3])
        QRS_duration_all.append(features[4])
        T_duration_all.append(features[5])
        PR_all.append(features[6])
        QT_all.append(features[7])
        ST_all.append(features[8])
        PT_all.append(features[9])
        STT_all.append(features[10])
        RP_peak_all.append(features[11])
        RR_all.append(features[12])
        QTc_all.append(features[13])
        TpTe_all.append(features[14])
        iCEB_all.append(features[15])
    #     print(f'Subject_ID: {rec_list2[i][0]}')

        #print(i,'= = = = = = = = = = = = = =\n')

    return parameters(data_in,p_peaks_all,r_peaks_all,t_peaks_all,P_duration_all,QRS_duration_all,T_duration_all,PR_all,QT_all,ST_all,PT_all,STT_all,RP_peak_all,RR_all,QTc_all,TpTe_all,iCEB_all)

