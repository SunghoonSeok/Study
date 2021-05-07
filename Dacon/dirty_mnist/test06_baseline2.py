# Library Load
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from torch_poly_lr_decay import PolynomialLRDecay
import random
import ttach as tta
import albumentations

torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_df = pd.read_csv('c:/data/test/dirty_mnist/dirty_mnist_2nd/*')[:]
imgs_dir = np.array(sorted(glob.glob('c:/data/test/dirty_mnist/dirty_mnist_2nd/*')))[:]
labels = np.array(labels_df.values[:,1:])
test_imgs_dir = np.array(sorted(glob.glob('c:/data/test/dirty_mnist/test_dirty_mnist_2nd/*')))

imgs=[]
for path in tqdm(imgs_dir[:]): # tqdm은 for문 상태바라고 생각하면 됨
    img = cv2.imread(path, cv2.IMREAD_COLOR) # 이미지 경로와 컬러 흑백 선택
    imgs.append(img)
imgs = np.array(imgs)

class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass # 지나쳐 가기 위한 코드
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        pass




#     def __getitem__(self, idx):
#         # 1개 샘플 get
#         img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
#         img = self.transform(img)
#         if self.train==True:
#             label = self.labels[idx]
#             return img, label
#         else:
#             return img
        
#         pass
    