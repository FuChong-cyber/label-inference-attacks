# label-inference-attacks
Code &amp; supplementary material of the paper Label Inference Attacks Against Federated Learning on Usenix Security 2022.

## Prerequisites
Install Python 3.8 and Pytorch 1.7.0 +

## Dataset Setup
### Dataset Download
Download the following datasets to './Code/datasets'.

CINIC-10 [1]

Yahoo answers dataset:

https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset

Criteo dataset:

https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/

Breast histopathology images: 

https://www.kaggle.com/paultimothymooney/breasthistopathology-images

Tiny ImageNet:

https://www.kaggle.com/c/tiny-imagenet

Breast cancer wisconsin dataset:

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

CIFAR-10 or CIFAR-100:

Use pytorch built-in classes.

### Dataset Preprocess
Use scripts in './Code/datasets_preprocess' to preprocess the datasets.

## Quick Start
### For Windows OS
Use batch files in the './Code' folder.

'run_training.bat':
train simulated VFL models.

'run_model_completion.bat':
run the passive and active label inference attacks.

'run_direct_attack.bat':
run the direct label inference attack.

'run_training_possible_defense.bat':
test possible defenses against the passive and active label inference attacks.

'run_direct_attack_possible_defense.bat':
test possible defenses against the direct label inference attack.
### For Linux OS
Use commands in the batch files, e.g., use commands in 'run_training.bat' to train simulated VFL models.

## References
[1] L. N. Darlow, E. J. Crowley, A. Antoniou, and A. J.
Storkey. CINIC-10 is not ImageNet or CIFAR-10. arXiv
preprint arXiv:1810.03505, 2018.