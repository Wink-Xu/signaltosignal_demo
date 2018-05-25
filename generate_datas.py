import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import pandas as pd 

#plt.rcParams['figure.figsize'] = (10, 12.0)

NUM_HADAMARD = 64

#def normalize(y):
#    a,b=np.shape(y)
#    n = np.zeros((a,b))
#    for i in range(a):
#        for j in range(b):
#            n[i,j] = np.int(y[i,j])
#    return n


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


#rand_pattern = np.random.randn(DIM,DIM)
signals = []
for i in range(40000):
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

#sig = np.squeeze(signals)
data_full_samples = pd.DataFrame(data = signals)
data_full_samples.to_csv('F:/Code/signal-signal_fromscratch/data/data_full_samples.csv')

print("Signals have been written in CSV")


#**********************************************
#**********************************************
#**********************************************#**********************************************
#**********************************************#**********************************************
#**********************************************#**********************************************
#**********************************************#**********************************************
#**********************************************#**********************************************


x = pd.read_csv('F:/Code/signal-signal_fromscratch/data/data_full_samples.csv')
original_signals =np.array(x)
original_signals =original_signals[:,1:]
print("We get the original data")

signals_full_sample = np.abs(original_signals)


def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))*255

signals_normalized = []
for i in range(signals_full_sample.shape[0]):
    one_of_signals = normalize(signals_full_sample[i])
    signals_normalized.append(one_of_signals)
    

signals_normalized = np.array(signals_normalized)
print(signals_normalized.shape)

print("Signals normalize have been finished ")
preSort_signals = np.sum(signals_normalized,axis=0)

#signal_abs = np.abs(np.squeeze(signal))
signal_dict = {}
j= 0
for i in preSort_signals:
    j = j+1
    signal_dict[j] = i 

xxx =sorted(signal_dict.items(),key = lambda items:items[1],reverse = True)
collect_sorted_signal= [i[0] for i in xxx]
print("We get the collective sorted picture index !")

haha = pd.DataFrame(collect_sorted_signal)
haha.to_csv('F:/Code/signal-signal_fromscratch/data/collect_sorted_index.csv')

collect_sorted_signal1 = [x-1 for x in collect_sorted_signal]
sorted_signals =original_signals[:,collect_sorted_signal1]
sorted_signals_pd = pd.DataFrame(sorted_signals)
sorted_signals_pd.to_csv('F:/Code/signal-signal_fromscratch/data/sorted_signals.csv')