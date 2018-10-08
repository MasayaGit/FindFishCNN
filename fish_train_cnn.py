
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

#分類数
num_classes = 2
image_size = (32, 32)
#CNN作るときに下のやつ使っている
im_rows = 32
im_cols = 32
in_shape = (im_rows, im_cols, 3)

#os.getcwd() カレントディレクトリ取得
path= os.getcwd()
path_fish = path + '/fish'
path_nofish = path + '/nofish'


numx = []# 画像データ
numy = [] # ラベルデータ

# 画像データを読み込んで配列に追加 --- (*1)
files = glob.glob(path_nofish + "/*.jpg")
for f in files:
    im = cv2.imread(f)
    #色空間を変換してリサイズ
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, image_size)
    # 配列の変形とデータの正規化
    im = im.reshape(32,32,3).astype('float32')/255
    numx.append(im)
    numy.append(keras.utils.to_categorical(0, num_classes))


# 画像データを読み込んで配列に追加 --- (*1)
files = glob.glob(path_fish + "/*.jpg")
for f in files:
    im = cv2.imread(f)
    #色空間を変換してリサイズ
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, image_size)
    # 配列の変形とデータの正規化
    im = im.reshape(32,32,3).astype('float32')/255
    #データの追加をx,yにする。labelは引数で受け取っている。
    numx.append(im)
    numy.append(keras.utils.to_categorical(1, num_classes))

x = np.array(numx)
y = np.array(numy)



# 学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# モデルを定義 
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# モデルをコンパイル --- (*4)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 学習を実行
hist = model.fit(x_train, y_train,
    batch_size=32, epochs=50,
    verbose=1,
    validation_data=(x_test, y_test))



# モデルを評価 --- (*6)
score = model.evaluate(x_test, y_test, verbose=1)
print('正解率=', score[1], 'loss=', score[0])


model.save('fish_find_cnn.h5')

