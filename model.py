import tensorflow as tf
import numpy as np



def inference(input_data):
    with tf.variable_scope('fc1') as scope:
    #    X = tf.placeholder(tf.float32,[1 256],name = 'X1')
        weights = tf.get_variable('w1',
                              shape = [256,1024],
                              dtype = tf.float32,
                              initializer = tf.truncated_normal_initializer(stddev=0.03,dtype=tf.float32))
        bias = tf.get_variable('b1',shape=[1024],dtype = tf.float32 ,initializer = tf.constant_initializer(0.1))
        Y = tf.add(tf.matmul(input_data,weights),bias)
        output1 = tf.nn.relu(Y,name = scope.name)
    
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable(name = 'w2',
                                  shape = [1024,1024],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.03,dtype = tf.float32))
        bias = tf.get_variable(name ='b2',
                              shape = [1024],
                              dtype = tf.float32,
                              initializer = tf.constant_initializer(0.1))
        output2 = tf.nn.relu(tf.add(tf.matmul(output1,weights),bias),name = scope.name)
        
    with tf.variable_scope('fc3') as scope:
        weights = tf.get_variable(name = 'w3',
                                  shape = [1024,256],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.03,dtype = tf.float32))
        bias = tf.get_variable(name ='b3',
                              shape = [256],
                              dtype = tf.float32,
                              initializer = tf.constant_initializer(0.1))
        output2 = tf.add(tf.matmul(output1,weights),bias,name = scope.name)
  
    return output2