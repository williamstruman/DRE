# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:24:00 2017
We can only predict the future of the next minutes because our target_vector_sz is only 2
We will implement target_vector size to be 8 for the draw1_3

we change the draw1_6 to have input with indicators. 

@author: xiao7
"""


import pickle
import tensorflow as tf
import numpy as np
import os
import sys

import tensorflow.contrib.slim as slim

crd = 11
crd_original = 8
sl= 60
batch_size=8  
path = './saved_model/' 

read_size = crd
hidden_size_enc = 256       # hidden size of the encoder
hidden_size_dec = 256       # hidden size of the decoder
num_l = 10 # dimensionality of the latent space
                      # Sequence length
max_iterations=40100
learning_rate=1e-5
dropout = 0.8
eps=1e-8                    # Small number to prevent numerical instability

target_vector_sz = 2


REUSE_T=None               # indicator to reuse variables. See comments below
read_params = []           # A list where we save al read paramaters trhoughout code. Allows for nice visualizations at the end
write_params = []          # A list to save write parameters


# our read function just need to read the input of the x at the corrosponding step

def encode(state,input):
  #Run one step of the encoder
  with tf.variable_scope("encoder",reuse=REUSE_T):
    return lstm_enc(input,state)


with tf.variable_scope("placeholders") as scope:
#  x = tf.placeholder(tf.float32,shape=(batch_size,cv_size)) # input (batch_size * cv_size)
  
  x = tf.placeholder(tf.float32, shape=[batch_size,crd,sl], name = 'Input_data')
  x_next = tf.subtract(x[:,:target_vector_sz,1:], x[:,:target_vector_sz,:sl-1])  # this is just price difference, which we can also d oin terms of returns. 

  keep_prob = tf.placeholder("float")
  lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_size_enc) # encoder Op
  lstm_enc = tf.nn.rnn_cell.DropoutWrapper(lstm_enc,output_keep_prob = keep_prob)

  
with tf.variable_scope("States") as scope:
  canvas=[0]*sl
  enc_state=lstm_enc.zero_state(batch_size, tf.float32)


with tf.variable_scope("DRAW") as scope:
  for t in range(sl):
    r = x[:,:,t]
    h_enc,enc_state=encode(enc_state,r) # here r is input and enc_state is the state. 
    canvas[t] = h_enc
    REUSE_T=True
  
  outputs = canvas

with tf.name_scope("Loss") as scope:
#  outputs_tensor = tf.concat(0,outputs[:-1])
  outputs_target = outputs[:-1]
  W_o = tf.Variable(tf.random_normal(
          [hidden_size_enc, target_vector_sz], stddev=0.01))
  b_o = tf.Variable(tf.constant(0.5, shape=[target_vector_sz]))
  
  h_out_tensor = [tf.nn.xw_plus_b(out, W_o, b_o) for out in outputs_target]
  h_out = tf.stack(h_out_tensor,2)
  target_vector = x_next

  cost = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(h_out, target_vector))))


with tf.variable_scope("Optimization") as scope:
  global_step = tf.Variable(0,trainable=False)
  lr = tf.train.exponential_decay(learning_rate,global_step,14000,0.95,staircase=True)
  optimizer=tf.train.AdamOptimizer(lr)
  grads=optimizer.compute_gradients(cost)
  for i,(g,v) in enumerate(grads):  #g = gradient, v = variable
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # Clip the gradients of LSTM
        print(v.name)
  train_op=optimizer.apply_gradients(grads,global_step = global_step)

print('Finished comp graph')
#%%
saver = tf.train.Saver(max_to_keep=3);  

fetches=[cost_recon,cost_lat,train_op]
costs_recon=[0]*max_iterations
costs_lat=[0]*max_iterations


sess=tf.Session()
load_model = True
sess.run(tf.global_variables_initializer())

if load_model == True:
    print ('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        print('there are stuff')
#        ckpt.model_checkpoint_path = './saved_model/sm.ckpt-52915'
#        ckpt.all_model_checkpoint_paths[0] = './saved_model/sm.ckpt-52915'
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('model loaded')

#%%
max_iterations = 60000
for i in range(max_iterations):
  num_batch = i%4
  xtrain = input_batches[num_batch]
  xtrain = add_indicator(zero_xtrain(xtrain))
  xtrain = np.array(xtrain)
  x_train_obsv = xtrain[:,:crd_original,:int(sl)]
  x_train_forerror = xtrain[:,:,int(sl):int(sl*2)]
  
  x_train_obsv = np.reshape(x_train_obsv,(batch_size,crd_original*sl))
  
  feed_dict={x:x_train_forerror,keep_prob: dropout,x_obsv: x_train_obsv }
  results=sess.run(fetches,feed_dict)
  costs_recon[i],costs_lat[i],_=results
  if i%100==0:
    print("iter=%d : cost_recon: %f cost_lat: %f" % (i,costs_recon[i],costs_lat[i]))
  if i%1000 == 0:
    saver.save(sess,'.\saved_model\sm1.ckpt', global_step=i)
    print('model saved')
    print('\n')

print('this is the end of the experiment')


  
  
