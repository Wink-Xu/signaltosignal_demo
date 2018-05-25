import tensorflow as tf
import numpy as np
from train import *
from utils import *


#plt.rcParams['figure.figsize'] = (10, 12.0)

NUM_HADAMARD = 256

sort_index = pd.read_csv('F:/Code/signal-signal_fromscratch/data/collect_sorted_index.csv')
sort_index = np.array(sort_index)
sort_index = sort_index[:,1:]
sort_index = [x-1 for x in sort_index]

RD_dir = u'C:/Users/zh/Desktop/hadamard/32RD.png'
RD=Image.open(RD_dir)
RD = np.array(RD)
CC = np.ones((1024,1024))
RD_p = 2*RD - CC
RD_anti = -1 * RD_p

x = load_data('F:/Code/signal-signal_fromscratch/data/sorted_noise_signals.csv')
y = load_data('F:/Code/signal-signal_fromscratch/data/sorted_signals.csv')

x_train_data = x[:39895,:256]
y_train_data = y[:39895,:256]
x_test_data = x[39895:39995,:256]
y_test_data = y[39895:39995,:256]
x_eval_data = x[39995:,:256]
y_eval_data = y[39995:,:256]
BATCH_SIZE = 64 



if __name__=='__main__':
    train(x_train_data,y_train_data,x_eval_data,y_eval_data,sort_index,RD_p,epoch = 50)
    print("The train has been done . -_-")