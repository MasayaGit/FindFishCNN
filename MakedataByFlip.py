
# coding: utf-8

# In[1]:


import cv2
import os, glob
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


path_fish = '/home/FindFish/fish/'
path_nofish = '/home/FindFish/nofish/'



# 画像データを読み込んで反転させ画像生成 --- (*1)
def make_img(data_path):
    no = 10000
    files = glob.glob(data_path + '*.jpg')
    print("ok")
    for f in files:
        img = cv2.imread(f)
        img2 = cv2.flip(img,1)
        outfile = str(no) + ".jpg"
        cv2.imwrite(data_path + outfile,img2)
        no += 1

# 画像データを生成
make_img(path_nofish)
make_img(path_fish)

