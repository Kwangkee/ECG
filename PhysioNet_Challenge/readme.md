PhysioNet Challenges, https://moody-challenge.physionet.org/
- Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021, https://moody-challenge.physionet.org/2021/
- Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020, https://moody-challenge.physionet.org/2020/
- AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017, https://moody-challenge.physionet.org/2017/xyz


***
## PhysioNet Challenge 2021
- Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021, https://moody-challenge.physionet.org/2021/
- The PhysioNet/Computing in Cardiology Challenge 2021, https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Will+two+do%3F+Varying+dimensions+in+electrocardiography%3A+the+PhysioNet%2FComputing+in+Cardiology+Challenge+2021&btnG=

The final scores & ranking, https://moody-challenge.physionet.org/2021/results/
- 1st: Classification of ECG Using Ensemble of Residual CNNs with Attention Mechanism, https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Classification+of+ECG+Using+Ensemble+of+Residual+CNNs+with+Attention+Mechanism&btnG=
- 2nd: Towards High Generalization Performance on Electrocardiogram Classification, https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Towards+High+Generalization+Performance+on+Electrocardiogram+Classification&btnG=, [SNU, Huino] https://ieeexplore.ieee.org/abstract/document/9662737/authors#authors
- 3rd: Multi-label Cardiac Abnormality Classification from Electrocardiogram using Deep Convolutional Neural Networks, https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Multi-label+Cardiac+Abnormality+Classification+from+Electrocardiogram+using+Deep+Convolutional+Neural+Networks&btnG=

### Data
The training data contains twelve-lead ECGs. The validation and test data contains twelve-lead, six-lead, four-lead, three-lead, and two-lead ECGs:
- Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- Six leads: I, II, III, aVR, aVL, aVF
- Four leads: I, II, III, V2
 -Three leads: I, II, V2
- Two leads: I, II

### Evaluation Code & Metrics
PhysioNet/CinC Challenge 2021 Evaluation Metrics, https://github.com/physionetchallenges/evaluation-2021

Each ECG recording has one or more labels that describe cardiac abnormalities (and/or a normal sinus rhythm). We mapped the labels for each recording to SNOMED-CT codes. The lists of scored labels and unscored labels are given with the evaluation code; see the scoring section for details.


https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv

### 
Python example code for the PhysioNet/Computing in Cardiology Challenge 2021, https://github.com/physionetchallenges/python-classifier-2021