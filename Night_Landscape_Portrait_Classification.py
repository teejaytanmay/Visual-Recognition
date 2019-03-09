import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

def single_ch_hist(image, channels, bins, chrange, color):
    hist = cv2.calcHist(image, channels, None, bins, chrange)
    return hist

def plot_hist(image,bins,r1,r2,r3):
    histarr = []
    histarr.append(single_ch_hist(image, [0], [bins], [0, r1], 'r'))
    histarr.append(single_ch_hist(image, [1], [bins], [0, r2], 'g'))
    histarr.append(single_ch_hist(image, [2], [bins], [0, r3], 'y'))
    histarr = np.asarray(histarr)
    histarr = histarr.reshape((3, bins))
    plt.show()
    return histarr

folderarr = ["landscape","night","portrait"]
finarr = []
for i in folderarr:
    for j in range(1,44):
        img = cv2.imread(i + "/" + str(j) + ".jpg");
        print(i + "/" + str(j) + ".jpg")
        img_new = cv2.resize(img, (500, 500));
        img_hsv = cv2.cvtColor(img_new, cv2.COLOR_BGR2HSV)
        arr = plot_hist(img_hsv,32,179,255,255)
        arr1 = plot_hist(img_new,32,255,255,255)
        finarr.append(arr) #Adding histogram of HSV image
        finarr.append(arr1)#Adding histogram of BGR image
z = np.asarray(finarr)
z = z.reshape(129, 192)
features = pd.DataFrame(z)
features.columns = [str(col) + '_col' for col in data.columns]

labels = []
for i in folderarr:
    for j in range(43):
        labels.append((i))

label = pd.DataFrame(labels, columns=["labels"])
data = pd.concat([features, label], axis = 1)
data = shuffle(data)

X=data[
['0_col',
 '1_col',
 '2_col',
 '3_col',
 '4_col',
 '5_col',
 '6_col',
 '7_col',
 '8_col',
 '9_col',
 '10_col',
 '11_col',
 '12_col',
 '13_col',
 '14_col',
 '15_col',
 '16_col',
 '17_col',
 '18_col',
 '19_col',
 '20_col',
 '21_col',
 '22_col',
 '23_col',
 '24_col',
 '25_col',
 '26_col',
 '27_col',
 '28_col',
 '29_col',
 '30_col',
 '31_col',
 '32_col',
 '33_col',
 '34_col',
 '35_col',
 '36_col',
 '37_col',
 '38_col',
 '39_col',
 '40_col',
 '41_col',
 '42_col',
 '43_col',
 '44_col',
 '45_col',
 '46_col',
 '47_col',
 '48_col',
 '49_col',
 '50_col',
 '51_col',
 '52_col',
 '53_col',
 '54_col',
 '55_col',
 '56_col',
 '57_col',
 '58_col',
 '59_col',
 '60_col',
 '61_col',
 '62_col',
 '63_col',
 '64_col',
 '65_col',
 '66_col',
 '67_col',
 '68_col',
 '69_col',
 '70_col',
 '71_col',
 '72_col',
 '73_col',
 '74_col',
 '75_col',
 '76_col',
 '77_col',
 '78_col',
 '79_col',
 '80_col',
 '81_col',
 '82_col',
 '83_col',
 '84_col',
 '85_col',
 '86_col',
 '87_col',
 '88_col',
 '89_col',
 '90_col',
 '91_col',
 '92_col',
 '93_col',
 '94_col',
 '95_col',
 '96_col',
 '97_col',
 '98_col',
 '99_col',
 '100_col',
 '101_col',
 '102_col',
 '103_col',
 '104_col',
 '105_col',
 '106_col',
 '107_col',
 '108_col',
 '109_col',
 '110_col',
 '111_col',
 '112_col',
 '113_col',
 '114_col',
 '115_col',
 '116_col',
 '117_col',
 '118_col',
 '119_col',
 '120_col',
 '121_col',
 '122_col',
 '123_col',
 '124_col',
 '125_col',
 '126_col',
 '127_col',
 '128_col',
 '129_col',
 '130_col',
 '131_col',
 '132_col',
 '133_col',
 '134_col',
 '135_col',
 '136_col',
 '137_col',
 '138_col',
 '139_col',
 '140_col',
 '141_col',
 '142_col',
 '143_col',
 '144_col',
 '145_col',
 '146_col',
 '147_col',
 '148_col',
 '149_col',
 '150_col',
 '151_col',
 '152_col',
 '153_col',
 '154_col',
 '155_col',
 '156_col',
 '157_col',
 '158_col',
 '159_col',
 '160_col',
 '161_col',
 '162_col',
 '163_col',
 '164_col',
 '165_col',
 '166_col',
 '167_col',
 '168_col',
 '169_col',
 '170_col',
 '171_col',
 '172_col',
 '173_col',
 '174_col',
 '175_col',
 '176_col',
 '177_col',
 '178_col',
 '179_col',
 '180_col',
 '181_col',
 '182_col',
 '183_col',
 '184_col',
 '185_col',
 '186_col',
 '187_col',
 '188_col',
 '189_col',
 '190_col',
 '191_col'
]]

y=data['labels']

scores=[]
knn = KNeighborsClassifier(algorithm='auto', metric_params=None, n_jobs=None, n_neighbors=3,weights='uniform')
cv_score = np.mean(cross_val_score(knn, X, y, cv=6))
scores.append(cv_score)
print(scores)
