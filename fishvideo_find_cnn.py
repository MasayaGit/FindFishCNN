
# coding: utf-8

# In[1]:


import cv2, os, copy
from sklearn.externals import joblib
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json


model = load_model('fish_find_cnn.h5')

output_dir = "./bestshot"
img_last = None # 前回の画像
img_th = 2 # 画像を出力するかどうかのしきい値。魚が何匹以上映っているか？ 
count = 0
frame_count = 0
#魚が映っているのかの閾値
fish_threshold = 0.7
if not os.path.isdir(output_dir): os.mkdir(output_dir)

image_size = (32, 32)
 
    
# 動画ファイルから入力を開始 
cap = cv2.VideoCapture("fish1.mp4")
while True:
    # 画像を取得
    is_ok, frame = cap.read()
    if not is_ok: break
    frame = cv2.resize(frame, (640, 360))
    frame2 = copy.copy(frame)
    frame_count += 1
    # 前フレームと比較するために白黒に変換 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if not img_last is None:
        # 差分を得る
        frame_diff = cv2.absdiff(img_last, img_b)
        cnts = cv2.findContours(frame_diff, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[1]
        # 差分領域に魚が映っているか調べる
        fish_count = 0
        for pt in cnts:
            x, y, w, h = cv2.boundingRect(pt)
            if w < 100 or w > 500: continue # ノイズを除去
            # 抽出した領域に魚が映っているか確認 --- (*3)
            im = frame[y:y+h, x:x+w]
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, image_size)
            # 配列の変形とデータの正規化
            im = im.reshape(32,32,3).astype('float32')/255
            pred_y = model.predict(np.array([im]),batch_size=32,verbose=1) # --- (*4)
            #２次元配列　値２つ入っている。
            #魚がいるかのカウントを行う
            if pred_y[0][1] >= fish_threshold:
                fish_count += 1
                cv2.rectangle(frame2, (x, y), (x+w+10, y+h+10), (0,255,0), 2)
        # 魚が何匹以上映っているか？ ある数より映っていれば保存--- (*5)
        if fish_count >= img_th:
            fname = output_dir + "/fish" + str(count) + ".jpg"
            cv2.imwrite(fname, frame2)
            count += 1
    if cv2.waitKey(1) == 13: break
    img_last = img_b
cap.release()
cv2.destroyAllWindows()
print("ok", count, "/", frame_count)

