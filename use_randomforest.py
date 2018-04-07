#使用 隨機森林 (Random Forest) 方法

import pandas as pd #使用 pandas 讀檔和處理屬性下的值
import numpy as np #使用 numpy 計算陣列

from sklearn import metrics #主要使用 accuracy_score 函式 比對精確值
from sklearn.cross_validation import train_test_split #切分訓練用
from sklearn.ensemble import RandomForestClassifier #隨機森林
from sklearn.metrics import classification_report #顯示測驗結果

#讀取目標檔案
csv = pd.read_csv('/home/f74046577/Database/winequality-red_creative.csv')

#建立空陣列
arr = []

#對 'quality' 屬性下所有的數值轉為數值區間
for x in csv['quality']:
    if x < 5:
        x = 1 # 0 ~ 4 屬於 低品質
    elif x > 6:
        x = 10 # 7 ~ 10 屬於 高品質
    else:
        x = 5 # 5~6 屬於中間品質
    arr.append(x) #將處理過的值丟到陣列裡面

#將 'quality' 屬性下所有數值替換為arr的數值
csv['quality'] = arr

# 初始化陣列
arr = []
# 這裡是創意新增的攔位 (雖然也是瞎掰的)
# taste 人吃過給的評價 0~10 分
for x in csv['taste']:
    if x < 4:
        x = 'bad' # <4 bad
    elif x > 8:
        x = 'excellent' # > 8 excellent
    else:
        x = 'normal' # 5 ~ 8 normal
    arr.append(x) #將改變後的元素加入陣列

#取代taste屬性下所有的值
csv['taste'] = arr

#切分訓練 x 對 y
x = csv[['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide',\
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']]
#切分訓練目標屬性
y = csv[['taste']]

#切分訓練開始 (test_size 設為 0.3, train_size 設為 0.7, random_state 使用 None) 
#回傳值放在 x_train, x_test, y_train, y_test 裡
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=20180407)

#引入StandardScaler讓變數標準化。
#避免偏向特定屬性做訓練
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#先對 x_train 做聚類
#利用資料中心和縮放比例做標準化
#標準化後放入 x_train_nor 和 x_test_nor
sc.fit(x_train)
x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)

#使用隨機森林做處理，調適後最好的深度是12。
clf = RandomForestClassifier(criterion='entropy', max_depth=12) 
clf_fit=clf.fit(x_train_nor, y_train.values.ravel()) #values.ravel() 避免waring

#預測結果放入y_pred
y_pred = clf_fit.predict(x_test_nor)

#驗證資料準確性
accuracy = metrics.accuracy_score(y_test, y_pred)

#輸出結果
print(classification_report(y_test, y_pred))
print('-----------')
print('accuracy : ', accuracy)
