# Machine learning for classification of advanced rheumatic heart disease using electrocardiogram in cardiology ward
 
Machine learning for classification of advanced rheumatic heart disease from wearable electrocardiogram signals: link to paper (BMC Cardiovascular Disorders)

This repo contains code snippets used for the aforementioned paper. In this work we apply classical machine learning models and CNN models to perfrom detection of RHD from healthy subjects.
The work flow block diagram is shown below.

![WorkFlow](https://github.com/user-attachments/assets/04665597-70f2-4673-9713-d6244f917551)

## Dataset

We use the age-matched subjects having Normal sinus rhythm from PTB-XL database from the [PhysioNet/CinC PTB-XL database](https://physionet.org/content/ptb-xl/1.0.3/), which contains in total of 6996 ECG records. In addition, RHDECG dataset that consists of 146 subjects, of which 117 subjects with RHD, and the remaining were healthy controls. 

## Validation metrics
<img src="https://github.com/user-attachments/assets/01475718-f306-422d-b391-1ad0335abfd4" width="300" height="300">

```bash
Accuracy = (TP + TN) \ (TP + FN + FP + TN)
Sensitivity = TP \ (TP + FN) 
Specificity = TN \ (FP + TN) 
F1_Score = 2 * TP \ (2 * TP + FP + FN)
```

## Results

The extracted time-frequency features and raw ECG of 10-second duration were used for classification. We evaluated different experiments and the obtained results suggest potential use of ECGs in RHD detection, helping in reduction of disease intervention burden particularly at resource onstrained medical setting.
![image](https://github.com/user-attachments/assets/ff9769d5-e7e5-448c-95f5-5a7435865829)
![image](https://github.com/user-attachments/assets/b1759514-fe10-43f6-a9f6-654e36c7e666)

![image](https://github.com/user-attachments/assets/89854268-da68-403f-ab19-f2ba4c3415ab)

<!--- comment 
10 fold validation with model CNN

|Fold      |Accuracy|Sensitivity|Specificity|MAcc   | F1 Score  |
| ----------|--------|---------- | ----------|-------| ----------|
|1          |0.928   |0.891      |0.925      |0.908  |0.933      |
|2          |0.941   |0.961      |0.866      |0.913  |0.963      |
|3          |0.917   |0.926      |0.881      |0.903  |0.946      |
|4          |0.929   |0.914      |0.985      |0.950  |0.953      |
|5          |0.935   |0.946      |0.896      |0.921  |0.959      |
|6          |0.939   |0.775      |0.940      |0.857  |0.866      |
|7          |0.986   |0.876      |0.924      |0.900  |0.924      |
|8          |0.948   |0.888      |0.940      |0.913  |0.933      |
|9          |0.918   |0.899      |0.894      |0.897  |0.934      |
|10         |0.924   |0.888      |0.848      |0.868  |0.922      |
|Average    |0.969   |0.896      |0.910      |0.903  |0.933      |  
--->
## Requirements

```bash
h5py==2.10.0
numpy==1.20.3
pandas==1.3.3
scikit-learn==0.24.2
scipy==1.7.1
spacy==3.2.1
spacy-legacy==3.0.8
spacy-loggers==1.0.1
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tokenizers==0.11.4
tqdm==4.62.2
urllib3==1.26.6
zsvision==0.7.12
neurokit==0.2.10
hrv-analysis==1.0.4
```
