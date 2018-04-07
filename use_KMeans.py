#使用 K-Means 方法

import pandas as pd # 使用 pandas 讀檔和處理屬性下的值
import numpy as np #使用 numpy 計算陣列
from sklearn.cluster import KMeans # 使用 KMeans 函式做分群

#讀入目標檔案
csv = pd.read_csv('/home/f74046577/Database/movielen10k.csv')
csv2 = pd.read_csv('/home/f74046577/Database/movielen10k_creative.csv')

#刪除我認為不需要的屬性
del csv['User Gender'], csv['User ZIP'], csv['Release Date'], csv['Rating'],\
csv['Release Date.year'], csv['Release Date.month'], csv['Release Date.day-of-month'],\
csv['Release Date.day-of-week'], csv['User Occupation']
#同上。
del csv2['genre']

#建立空陣列
arr = []

#對 'User age' 屬性下所有值做區間 
for x in csv['User age']:
    if x > 0 and x <= 10: # 1 ~ 10 為 區間 5 以下類推
        x = 5
    elif x > 10 and x <= 20:
        x = 15
    elif x > 20 and x <= 30:
        x = 25
    elif x > 30 and x <= 40:
        x = 35
    elif x > 40 and x <= 50:
        x = 45
    elif x > 50 and x <= 60:
        x = 55
    elif x > 60 and x <= 70:
        x = 65
    else:
        x = 75
    arr.append(x) #將改變後的元素加入陣列

#取代 'User age' 屬性下所有的值
csv['User age'] = arr

#除了 'User age', 'User Occupation' 屬性之外
#對所有屬性下的布林(bool)換成數值
for x in csv:
    arr = [] #首先將陣列初始化
    if x != 'User age' and x != 'User Occupation':
        for X in csv[x]:
            if X :
                X = 1 #True = 1
            else:
                X = 0 #False = 0 
            arr.append(X) #將改變後的元素加入陣列
        #取代當前屬性下所有的值
        csv[x] = arr

#前處理完畢
#合併2個資料集
merged_csv = pd.concat([csv, csv2], axis=1 )
    
#建立一個字典，用於數值轉字串
#這裡用在推薦項目轉換
#0 是 Comedy ... 類推
dict2 = {
    0 : 'Comedy',
    1 : 'Honnor',
    2 : 'Adventure',
    3 : 'Fantasy',
    4 : 'Active'
}

#把csv檔變成矩陣(nparray)
matrix = csv.as_matrix()
#使用KMeans分群
#並將分群結果放進clusters變數中
clusters = KMeans(n_clusters = 5).fit_predict(matrix)

#陣列初始化
arr = []

#將分群結果加總 (0 到 5)
for i in range(0, 5):
    x = np.sum(clusters == i) # x 轉換成 i 群總和
    arr.append(x) # 把總和加入陣列

#印出各族群總和
print(sorted(arr))

#陣列初始化
arr = []

#使用先前建立的字典
#將每個群轉換成各自的推薦目錄
for x in clusters:
    x = dict2[x]  #用字典轉換 ex : x = dict[0] --> 輸出 x = 'Comedy'
    arr.append(x) #加入陣列

#用整理好的值取代clusters
clusters = arr

#建立一個新的dataframe (panda的函式)
res = pd.DataFrame(clusters, columns = ["commend_movie"])

#合併屬性攔
merged_csv = pd.concat([res, merged_csv], axis=1 )

#顯示結果
merged_csv
