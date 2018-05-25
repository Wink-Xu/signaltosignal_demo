import tensorflow as tf
import numpy as np
from utils import *
from model import inference

def train(Xdata , Ydata ,Xevaldata,Yevaldata,sort_index,RD_p, learning_rate = 0.0001, epoch = 20 , batch_size = 32):
    
#    tf.reset_default_graph()
    logs = r"F:/Code/signal-signal_fromscratch/logs/"
    ckpt = r"F:/Code/signal-signal_fromscratch/checkpoints/"
    sample_dir = r'F:/Code/signal-signal_fromscratch/data/evaluate'
    X = tf.placeholder(dtype = tf.float32, shape =[None,256] ,name = 'X')
    Y = tf.placeholder(name = 'Y', shape =[None,256] ,dtype = tf.float32)
    
    output = inference(X)
    
    losses = tf.reduce_mean(tf.square(output-Y))
    
    tf.summary.scalar('loss',losses)


    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(losses) 
    
    init = tf.global_variables_initializer()
    print("Start training ...")
    
    print("Please wait for a while ...")
    with tf.Session() as sess:
        
        sess.run(init)
        iter_num = 0
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logs,sess.graph)
        
        numBatch = np.int(Xdata.shape[0]/batch_size)
        for i in range(epoch):
            for batch_id in range(numBatch):
                batch_Xdata = Xdata[batch_id*batch_size:(batch_id+1)*batch_size,:]
                batch_Ydata = Ydata[batch_id*batch_size:(batch_id+1)*batch_size,:]
                _ , loss,summary = sess.run([train_op , losses,merged] , feed_dict ={X:batch_Xdata ,Y:batch_Ydata })
                iter_num += 1
                summary_writer.add_summary(summary,iter_num)
            if i%2 == 0:
                print("epoch %d loss %f " % (i,loss))
            if i%10 == 0:
                print("Evaluating ...")
      
                for j in range(Xevaldata.shape[0]):
                    data = [Xevaldata[j][:]]
                    eval_out = sess.run(output,feed_dict = {X:data})
                    
                    sum1 = np.zeros((32,32))
                    sum2 = np.zeros((32,32))
                    sum3 = np.zeros((32,32))
                    y = 0
                    for k in np.squeeze(eval_out):
                        y = y+1
                        pattern = RD_p[sort_index[y-1]].reshape((1024,1)) 
                        sum1 = sum1 + Xevaldata[j][y-1] * pattern.reshape((32,32))
                        sum2 = sum2 + Yevaldata[j][y-1] * pattern.reshape((32,32))
                        sum3 = sum3 + k * pattern.reshape((32,32))
                    sum1[0,0] = np.mean(sum1)  
                    sum2[0,0] = np.mean(sum2)  
                    sum3[0,0] = np.mean(sum3)  
          
                    save_image(os.path.join(sample_dir, 'test%d_%d.png' % (y, iter_num)),sum2,sum1,sum3)
            
                save_model(sess,ckpt,iter_num)
        