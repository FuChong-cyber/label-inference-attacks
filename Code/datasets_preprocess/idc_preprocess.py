import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from glob import glob
import fnmatch

pixel_fat_judge_value = 220
fat_ratio_thresh_hold = 0.3

csv_file = 'E:/Dataset/BC_IDC_muscle'
img_paths_list = glob(csv_file + '/**/*.png', recursive=True)
img_id = 0
img_num = len(img_paths_list)
current_progress = -1
for img_path in img_paths_list:
    img_id += 1
    progress = int(img_id / img_num * 100)
    if progress != current_progress:
        print("progress:", progress, "%")
        current_progress = progress
    image = cv2.imread(img_path)
    pixel_count = 50 * 50
    image_pixel_ave = np.average(image, axis=2).flatten()
    white_pixel_count = len(np.where(image_pixel_ave > pixel_fat_judge_value)[0])
    fat_ratio = white_pixel_count / pixel_count
    # print("fat_ratio:", fat_ratio)
    if fat_ratio > fat_ratio_thresh_hold:
        # cv2.imshow('input_image', image)
        # cv2.waitKey(0)
        os.remove(img_path)
