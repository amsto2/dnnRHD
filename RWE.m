close all;
clear all;
%% Load dataset
% x=load('datapath\RHD.mat');
% x = load('datapath\NSR.mat');
% x=load('datapath\physionet.mat');
% x=load('datapath\CHF.mat');
% x = load('datapath\PTB.mat');

% get the size
s=size(x.ECGRecord);

%% Compute relative wavelet energy
d_levels=10
energy=zeros(s(1),d_levels); % we decompose the signal into d_levels-1 levels
relative_energy=zeros(s(1),d_levels);
Shannon_energy=zeros(s(1),d_levels);
relative_Shanon_energy=zeros(s(1),d_levels);
for j=1:s(1)
    Input_Signal=x.ECGRecord(j,1:5000);  % read 10sec record at the sampling frequency
    Input_Signal = Input_Signal-mean(Input_Signal);
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Compue relative energy of a signal using MODWT
    % https://nl.mathworks.com/help/wavelet/ref/modwt.html
    level=d_levels-1;
    ecg_coef = modwtmra(Input_Signal,level,'db4')%,TimeAlign=true);
    sig_len=length(ecg_coef)
    energy_by_scales = 1/sig_len * sum(ecg_coef.^2,2);
    Levels = {'D1';'D2';'D3';'D4';'D5';'D6';'D7';'D8';'D9';'A9'};
    energy_total = sum(energy_by_scales);
    relative_energy_by_scales = energy_by_scales./sum(energy_by_scales) ; % take percentages of relative energy
    shannon_energy=-1*abs(energy_by_scales).*log2(abs(energy_by_scales));
    relative_shannon_energy = shannon_energy./sum(shannon_energy); 
    table(Levels,energy_by_scales,relative_energy_by_scales,relative_shannon_energy)
    
    energy(j,:)=energy_by_scales;
    relative_energy(j,:)=relative_energy_by_scales;
    Shannon_energy(j,:) = shannon_energy;
    relative_Shanon_energy(j,:) = relative_shannon_energy;
     
end
%% Save file to path
% SE = -|a[n]|log(|a[n]|) %-1*abs(energy_by_scales).*log2(abs(energy_by_scales));
% save('datapath\relative_energy_nsr_final', 'relative_energy', '-v7.3')
% save('datapath\relative_Shanon_energy_NSR', 'relative_Shanon_energy', '-v7.3')
% save('datapath\relative_Shanon_energy_RHD', 'relative_Shanon_energy', '-v7.3')
% save('datapath\relative_energy_RHD9', 'relative_energy', '-v7.3')
% save('datapath\relative_energy_physiodata', 'relative_energy', '-v7.3')
% save('datapath\relative_energy_PTB_test', 'relative_energy', '-v7.3')
