import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from adaboost import Adaboost
from segment import segment
from make_feature import make_feature
import pickle

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/ len(y_true)
    return accuracy

# xy_data1 = np.loadtxt('ball.dat')
# rows_per_second = 720
# num_columns = 2
# tol_t = xy_data1.shape[0] // rows_per_second
# # 重塑数据为三维数组
# xy_data = xy_data1.reshape(tol_t, rows_per_second, num_columns).transpose(1, 2, 0)
# # # reshaped_data 的维度为 (rows_per_second, num_columns, total_seconds)
# size_xy = xy_data.shape
# # 初始化训练集

xy_data=loadmat('./box1.mat')['box1']
#print(xy_data.keys())
# print(data[:,:,0])
a,b ,tol_t = xy_data.shape
# print(tol_t)
train_X = []
# train_Y = np.loadtxt('ball_label.txt', unpack=True)
train_Y=loadmat('./trainybox.mat')['train_Y']
#print(train_Y.keys())
# 循环处理每个时间片
for sec in range(1,tol_t+1):
    # 调用 Segment 函数
    Seg, Si_n, S_n = segment(xy_data[:, :, sec-1])

    # 循环处理每个分段
    for i in range(S_n):
        # 调用 make_feature 函数并将结果添加到 train_X
        seg = Seg[i][:Si_n[i]]
        # seg = Seg[:Si_n[i], i]
        train_X.append(make_feature(xy_data[seg, :, sec-1]))


# 转换为 NumPy 数组
train_X = np.array(train_X)
train_Y = np.array(train_Y)
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=5)

#Adaboost classification with 5 weak classifiers
# clf = Adaboost(n_clf=100)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
y_train = y_train.ravel()
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(x_train, y_train)
y_pred = ada_model.predict(x_test)

# acc = accuracy(y_test, y_pred)
# print("Accuracy", acc)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

with open('adaboost_box_model.pkl', 'wb') as model_file:
    pickle.dump(ada_model, model_file)