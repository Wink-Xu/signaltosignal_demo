import numpy as np
from PIL import Image
import matplotlib.pylab as plt

#plt.rcParams['figure.figsize'] = (10, 12.0)

NUM_HADAMARD = 64

def normalize(y):
    a,b=np.shape(y)
    n = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            n[i,j] = np.int(y[i,j])
    return n


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
i = np.random.randint(100)


#rand_pattern = np.random.randn(DIM,DIM)


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
#sum1 = normalize(((sum1 - np.min(sum1))/(np.max(sum1) - np.min(sum1)))*256)
#image_ghost = Image.fromarray(sum1)
#image_ghost = image_ghost.convert('L')
sum1[0,0] = np.mean(sum1)
plt.figure(figsize=(8,10))

plt.subplot(211)
plt.title("Sequence Signal",fontsize =18)
plt.xlabel("Hadamard Row")
plt.ylabel("Idensity")
plt.plot(np.squeeze(signal))

plt.subplot(212)
signal_sorted = sorted(np.abs(signal),reverse=True)
plt.title("Sorted Signal",fontsize =18)
plt.plot(np.squeeze(signal_sorted))

plt.tight_layout()
plt.show()

print("****************** ")
print("****************** ")
print("****************** ")

print("Get the bigger signal ... ")

signal_abs = np.abs(np.squeeze(signal))
signal_dict = {}
j= 0
for i in signal_abs:
    j = j+1
    signal_dict[j] = i 

	##  排序
xxx =sorted(signal_dict.items(),key = lambda items:items[1],reverse = True)
signal_picture = [i[0] for i in xxx]


sum2 = np.zeros((32,32))
sum3 = np.zeros((32,32))
signal = np.squeeze(signal)
for j in signal_picture[:NUM_HADAMARD]:
    RD_pattern = RD_p[j-1].reshape((1024,1))
    anti_pattern = RD_anti[j-1].reshape((1024,1))
    pattern = RD_p[j-1].reshape((1024,1)) 
    temp = np.dot(pic,RD_pattern) - np.dot(pic,anti_pattern)
    sum2 = sum2 + temp * pattern.reshape((32,32))
y = 0
for k in signal[:NUM_HADAMARD]:
    y= y+1
    pattern = RD_p[y-1].reshape((1024,1)) 
    sum3 = sum3 + k * pattern.reshape((32,32))
#sum1 = normalize(((sum1 - np.min(sum1))/(np.max(sum1) - np.min(sum1)))*256)
#image_ghost = Image.fromarray(sum1)
#image_ghost = image_ghost.convert('L')
sum2[0,0] = np.mean(sum2)
sum3[0,0] = np.mean(sum3)
plt.figure()

plt.subplot(131)
plt.title("Full Samples",fontsize =18)
plt.imshow(sum1,cmap = plt.cm.gray)
plt.axis('off')

plt.subplot(132)
plt.title("Bigger "+str(NUM_HADAMARD),fontsize =18)

plt.imshow(sum2,cmap = plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.title("First " +str(NUM_HADAMARD),fontsize =18)
plt.imshow(sum3,cmap = plt.cm.gray)
plt.axis('off')
plt.show()