import os
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

images = []
label_arr = []
for j in range(1,6):
    with open('../cifar-10-python/cifar-10-batches-py/' + 'data_batch_' + str(j), 'rb') as fl:
        data_load = pickle.load(fl, encoding='bytes')

    data = data_load[b'data']
    label_arr.append(data_load[b'labels'])
    for i in range(data.shape[0]):
        image = []
        r = data[i][0:1024].reshape(32, 32)
        g = data[i][1024:2048].reshape(32, 32)
        b = data[i][2048:3072].reshape(32, 32)
        image.append(r)
        image.append(g)
        image.append(b)
        images.append(image)
label_arr = np.array(label_arr)
label_arr = label_arr.flatten().tolist()

input_data = np.array(images)
des_arr = []
new_des = []
labels = []
count = 0
for i, image in enumerate(input_data, 0):
    trans_img = image.transpose()

#     sift = cv2.xfeatures2d.SIFT_create()
#     (kp, des) = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), None)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(cv2.cvtColor(trans_img,cv2.COLOR_BGR2GRAY), None)

    if (len(kp) != 0):
        labels.append(label_arr[i])
        des_arr.append(des)
        for desc in des:
            new_des.append(desc)

data = pd.DataFrame(data = new_des)
kmeans = MiniBatchKMeans(n_clusters = 100,random_state = 100,batch_size=100)
kmeans.fit(data)
img_words = []
hist_arr = []
bow_arr = []
for desc in des_arr:
    img_words.append(kmeans.predict(desc))
for clus in img_words:
    bow_arr.append(np.bincount(clus, minlength = 100))
hist_arr = np.array(bow_arr)

data = pd.DataFrame(data = hist_arr)
label_data = pd.DataFrame(data = labels)
label_data.columns = ['label']
final_data = pd.concat([data, label_data], axis = 1)

train, test = train_test_split(final_data, test_size = 0.2)
train_ind = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99]
labels = ['label']

clf1 = LinearSVC(random_state=0, tol=1e-5)
clf1.fit(train[train_ind], train[labels])
pred1 = clf1.predict(test[train_ind])
print("Accuracy of SVC is: "+str(accuracy_score(pred1, test[labels])))

clf2 = LogisticRegression(random_state=0,solver='sag',multi_class='multinomial')
clf2.fit(train[train_ind], train[labels])
pred2 = clf2.predict(test[train_ind])
print("Accuracy of LR is: "+str(accuracy_score(pred2, test[labels])))

clf3 = KNeighborsClassifier(n_neighbors = 2)
clf3.fit(train[train_ind], train[labels])
pred3 = clf3.predict(test[train_ind])
print("Accuracy of KNN is: "+str(accuracy_score(pred3, test[labels])))

# , 100, 101, 102, 103,
# 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
# 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
# 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
# 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
# 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
# 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
# 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
# 195, 196, 197, 198, 199
