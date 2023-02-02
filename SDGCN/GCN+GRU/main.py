# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_beijing_data
from tgcn import tgcnCell
from gru import GRUCell

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph_mode', 'dynamic', 'fix or dynamic')
flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_float('threshold', 0.001, 'threshold.')
flags.DEFINE_integer('training_epoch', 101, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units',256, 'hidden units of gru.')
flags.DEFINE_integer('seq_len',12, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.7, 'rate of training set.')
flags.DEFINE_integer('batch_size', 1, 'batch size.')
flags.DEFINE_string('dataset', 'beijing', 'chengdu or xian.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
flags.DEFINE_string('restore', 'no', 'yes or no')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
method = FLAGS.graph_mode

restore = FLAGS.restore


graph_mode = FLAGS.graph_mode
threshold = FLAGS.threshold
###### load data ######

if data_name == 'beijing':
    data, static_adj = load_beijing_data('beijing')

time_len = data.shape[0]

num_nodes = data.shape[1]
if graph_mode == 'fix':
    adj = static_adj
elif graph_mode == 'dynamic':
    adj = np.zeros([num_nodes,num_nodes])


data1 =np.mat(data,dtype=np.float32)

#### normalization
max_value = np.max(data1)

data1 = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)



def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states
        
###### placeholders ######
laplas = tf.placeholder(tf.float32, shape=[num_nodes, num_nodes])
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}

if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred
      

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_pre%r'%(model_name,data_name,pre_len)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()


    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
 
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
loss2 = []
rmse2 = []


for epoch in range(training_epoch):
    loss1 = []
    rmse1 = []
    for m in range(totalbatch):
        if graph_mode == 'fix':
            adj = static_adj
        elif graph_mode == 'dynamic':
            beijing_adj = pd.read_csv(r'data/beijing_adj/adj_' + str(seq_len + m ) + '.csv', header=None)
            adj = np.mat(beijing_adj)
            adj = np.array(adj)


        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1_t, rmse1_t, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch,laplas:adj, labels:mini_label})
        loss1.append(loss1_t)
        rmse1.append(rmse1_t * max_value)
    loss1 = np.sum(np.array(loss1)) / totalbatch
    rmse1 = np.sum(np.array(rmse1)) / totalbatch
    batch_loss.append(loss1)
    batch_rmse.append(rmse1)


    # Test completely at every epoch
    total_test_output = []
    test_label = []

    loss2_1 = []
    rmse2_1 = []

    for k in range(int(testX.shape[0]/batch_size)):
        if graph_mode == 'fix':
            adj = static_adj
        elif graph_mode == 'dynamic':
            beijing_adj = pd.read_csv(r'data/beijing_adj/adj_' + str(seq_len + m + k) + '.csv', header=None)
            adj = np.mat(beijing_adj)
            adj = np.array(adj)

         # Test completely at every epoch
        loss2_t, rmse2_t, test_output = sess.run([loss, error, y_pred],
                                             feed_dict = {inputs:testX[k * batch_size: (k+1) * batch_size], laplas:adj,labels:testY[k * batch_size: (k+1) * batch_size]})
        ff = testY[k * batch_size: (k + 1) * batch_size]

        ff = np.reshape(ff, [-1, num_nodes])

        test_label.append(ff)
        total_test_output.append(test_output)

        loss2_1.append(loss2_t)
        rmse2_1.append(rmse2_t * max_value)



    aa = np.array(loss2_1)
    bb = np.array(rmse2_1)
    loss2 = np.sum(aa) / int(testX.shape[0]/batch_size)
    rmse2 = np.sum(bb) / int(testX.shape[0]/batch_size)

    tto = np.array(total_test_output)
    tl = np.array(test_label)
    cc = tto.reshape((-1,num_nodes)) * max_value
    dd = tl.reshape((-1,num_nodes)) * max_value

    rmse, mae, acc, r2_score, var_score = evaluation(dd, cc)

    test_loss.append(loss2)
    test_rmse.append(rmse2)
    test_mae.append(mae)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(dd)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_var_score:{:.4}'.format(var_score),
          'test_mae:{:.4}'.format(mae),
          'test_rmse2:{:.4}'.format(rmse2)
          )

    if restore == 'yes':
        if epoch == 1:
            if method == 'dynamic':
                print('out/tgcn/tgcn_' + data_name +'_pre' + str(pre_len) + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
                saver.restore(sess, 'out/tgcn/tgcn_' + data_name + '_pre' + str(pre_len) + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
            elif method == 'fix':
              print('out/tgcn/tgcn_' + data_name + '_pre' + str(pre_len) + '/fix_' + data_name + '_pre' + str(pre_len)  + '/' + data_name + '_fix_pre' + str(pre_len))
              saver.restore(sess, 'out/tgcn/tgcn_' + data_name + '_pre' + str(pre_len) + '/fix_' + data_name + '_pre' + str(pre_len)  + '/' + data_name + '_fix_pre' + str(pre_len))

    elif restore == 'no':
        if epoch == training_epoch-1:
            if method == 'dynamic':
                saver.save(sess, path + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
            elif method == 'fix':
                print(path + '/fix' + data_name + '_pre' + str(pre_len) + '/' + data_name + '_fix_pre' + str(pre_len))
                saver.save(sess, path + '/fix_' + data_name + '_pre' + str(pre_len) + '/' + data_name + '_fix_pre' + str(pre_len))

time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv',index = False,header = False)
#plot_result(test_result,test_label1,path)
#plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])
