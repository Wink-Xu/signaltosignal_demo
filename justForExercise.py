import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import pandas as pd 

#plt.rcParams['figure.figsize'] = (10, 12.0)

NUM_HADAMARD = 64



RD_dir = u'C:/Users/zh/Desktop/hadamard/32RD.png'
RD=Image.open(RD_dir)
RD = np.array(RD)
CC = np.ones((1024,1024))
RD_p = 2*RD - CC
RD_anti = -1 * RD_p
#RD_n = -1 * RD_p
# K = 256
# RD_use =np.vstack((RD_p[0:K,:],RD_n[0:K,:]))  #创造需要使用的 Hadamard 矩阵
#NUM_PICTURE = 42000
#RATIO = 0.1
DIM = 1024 
#i = np.random.randint(100)

signals = []
for i in range(200):
    filename1 = 'F:\Code\kaggle\Digit Recognizer\image\image' + '%d' % i+ '.jpg'
    x = Image.open(filename1)
    image_test = x.resize((32,32))
    image_test = np.array(image_test)
    pic = image_test.reshape((1,1024))
    sum1 = np.zeros((32,32))
    signal = []
    for j in range(1024):
        RD_pattern = RD_p[j].reshape((1024,1))
        anti_pattern = RD_anti[j].reshape((1024,1))
        pattern = RD_p[j].reshape((1024,1)) 
        temp = np.dot(pic,RD_pattern) - np.dot(pic,anti_pattern)
        signal.append(temp)
        sum1 = sum1 + temp * pattern.reshape((32,32))
    if i%2000 == 0:
        print("Have finished %d signal" %i)
    signal = np.squeeze(signal)
    signals.append(signal)

plt.figure()
for i in range(200):
    sum3 = np.zeros((32,32))
    signal = signals[i]
    y=0
    for k in signal[:1024]:
        y= y+1
        pattern = RD_p[y-1].reshape((1024,1)) 
        sum3 = sum3 + k * pattern.reshape((32,32))
    sum3[0,0] = np.mean(sum3)
    plt.imshow(sum3,cmap = plt.cm.gray)
    plt.axis('off')
    plt.pause(2)
  #  plt.show()