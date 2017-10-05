#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:11:20 2017

@author: rbao
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pdb

##import mnist dataset
#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

data_train = pd.read_csv('mnist_train.csv', header = None)
train = data_train.iloc[:,1::].as_matrix()
train = train.astype(np.float32)

train_label = data_train.iloc[:,0]
label_matrix = np.zeros([60000,10])
for i in range(len(train_label)):
    if train_label[i] == 0:
        label_matrix[i,0] = 1
    elif train_label[i] == 1:
        label_matrix[i,1] = 1
    elif train_label[i] == 2:
        label_matrix[i,2] = 1        
    elif train_label[i] == 3:
        label_matrix[i,3] = 1    
    elif train_label[i] == 4:
        label_matrix[i,4] = 1
    elif train_label[i] == 5:
        label_matrix[i,5] = 1
    elif train_label[i] == 6:
        label_matrix[i,6] = 1        
    elif train_label[i] == 7:
        label_matrix[i,7] = 1        
    elif train_label[i] == 8:
        label_matrix[i,8] = 1        
    elif train_label[i] == 9:
        label_matrix[i,9] = 1        


        
        
data_test = pd.read_csv('mnist_test.csv', header = None)
test = data_test.iloc[:,1::].as_matrix()
test = test.astype(np.float32)

test_label = data_test.iloc[:,0]
label_matrix_test = np.zeros([10000,10])
for i in range(len(test_label)):
    if test_label[i] == 0:
        label_matrix_test[i,0] = 1
    elif test_label[i] == 1:
        label_matrix_test[i,1] = 1
    elif test_label[i] == 2:
        label_matrix_test[i,2] = 1        
    elif test_label[i] == 3:
        label_matrix_test[i,3] = 1    
    elif test_label[i] == 4:
        label_matrix_test[i,4] = 1
    elif test_label[i] == 5:
        label_matrix_test[i,5] = 1
    elif test_label[i] == 6:
        label_matrix_test[i,6] = 1        
    elif test_label[i] == 7:
        label_matrix_test[i,7] = 1        
    elif test_label[i] == 8:
        label_matrix_test[i,8] = 1        
    elif test_label[i] == 9:
        label_matrix_test[i,9] = 1     


#data_train = np.random.rand(50,4)
#data_train_label = np.zeros((50,2))

#data_test = np.random.rand(20,4)
#data_test_label = np.zeros((20,2))


#define constants
#unrolled through 28 time steps
time_steps=28 # 4
#hidden LSTM units
num_units=128 # 8
#rows of 28 pixels
n_input=28 # 1
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=10 # 2
#size of batch
batch_size=128 # 8

"""

"""

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    
    while iter<460:
        
        # pdb.set_trace()
        batch_x = train[batch_size*iter:batch_size*iter+batch_size,:]
        batch_x = batch_x.reshape((batch_size,time_steps,n_input))
        batch_x = batch_x.astype(np.float32)
        batch_y = label_matrix[batch_size*iter:batch_size*iter+batch_size,:]
        

        # batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    #calculating test accuracy
    test_data = test[:128].reshape((-1, time_steps, n_input))
    test_label = label_matrix_test[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))




