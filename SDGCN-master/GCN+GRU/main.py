# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data, load_chengdu_data, load_XiAn_data
from HMM2adj import load_adj

from gcn import gcnCell
import seaborn as sns


from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
import time

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')
time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('dataset', 'xian', 'xian or chengdu.')
flags.DEFINE_integer('training_epoch',1500, 'Number of epochs to train.')
flags.DEFINE_integer('pre_len',1, 'time length of prediction.')
flags.DEFINE_string('restore', 'no', 'yes or no')
flags.DEFINE_string('restore_epoch', '100', 'restore_epoch')
flags.DEFINE_string('method', 'fix', 'dynamic or fix')
flags.DEFINE_string('visualization', 'False', 'True or False')
flags.DEFINE_integer('gru_units', 256, 'hidden units of gru.')
flags.DEFINE_string('show_time', 'no', 'yes or no')

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
flags.DEFINE_float('train_rate', 0.7, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('model_name', 'sdgcn', 'sdgcn')

vision = FLAGS.visualization
method = FLAGS.method
restore = FLAGS.restore
restore_epoch = FLAGS.restore_epoch
show_time = FLAGS.show_time

model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
if data_name == 'xian':
    data, adj = load_XiAn_data('xian')
if data_name == 'chengdu':
    data, adj = load_chengdu_data('chengdu')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

whole_adj = load_adj(data_name)

if method == 'dynamic':
    Laplas = whole_adj[0]
elif method == 'fix':
    Laplas = adj

# input visualization
fig = plt.figure()
ax1 = fig.add_subplot()
x = np.linspace(1, time_len + 1, time_len)
y = data1[:, 5]
ax1.set_ylabel('Traffic speed(km/h)', fontsize='15')
ax1.set_xlabel('Time', fontsize='15')
# ax1.set_title("Percentage %",fontsize='10',loc="left")

# chengdu setting
if data_name == "chengdu":
    ax1.set_xticklabels([0, "2018-11-01", "", "2018-11-03", "", "2018-11-06", "", "2018-11-09"], fontsize=10)
    ax1.set_yticklabels([0, 15, ' ', 25, ' ', 35, ' ', 40, ' '], fontsize=15)

# xian setting
if data_name == "xian":
    ax1.set_xticklabels([0,"2018-11-01","2018-11-04","2018-11-08","2018-11-11","2018-11-15"], fontsize=10)
    ax1.set_yticklabels([0, 15, ' ', 25, ' ', 35, ' '], fontsize=15)

ax1.plot(x, y, 'b')
plt.title('Single Traffic flow speed of one week')

plt.show()

#### normalization
max_value = np.max(data1)
data1 = data1 / max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

traintotalbatch = int(trainX.shape[0] / batch_size)
testtotalbatch = int(testX.shape[0] / batch_size)

training_data_count = len(trainX)

def SDGCN(_X, _weights, _biases):
    ###
    cell_1 = gcnCell(gru_units, Laplas, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']

    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states

###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}

if model_name == 'sdgcn':
    pred, ttts, ttto = SDGCN(inputs, weights, biases)

y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
# sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s' % (model_name)
# out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var


x_axe, train_batch_loss, train_batch_rmse, train_batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

test_loss2, test_rmse2, = [], []

for epoch in range(training_epoch):
    time_trainning_start = time.time()
    for m in range(traintotalbatch):
        if method == 'dynamic':
            Laplas = whole_adj[m]
        elif method == 'fix':
            Laplas = adj

        mini_train_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_train_label = trainY[m * batch_size: (m + 1) * batch_size]

        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_train_batch, labels: mini_train_label})
        train_batch_loss.append(loss1)
        train_batch_rmse.append(rmse1 * max_value)

    time_trainning_end = time.time()
    test_test_output = []
    # Test completely at every epoch

    time_predict_start = time.time()
    for m in range(testtotalbatch):
        if method == 'dynamic':
            Laplas = whole_adj[traintotalbatch + m]
        elif method == 'fix':
            Laplas = adj

        mini_test_batch = testX[m * batch_size: (m + 1) * batch_size]
        mini_test_label = testY[m * batch_size: (m + 1) * batch_size]

        test_batch_loss2, test_batch_rmse2, test_batch_test_output = sess.run([loss, error, y_pred],
                                                                              feed_dict={inputs: mini_test_batch,
                                                                                         labels: mini_test_label})
        test_loss2.append(test_batch_loss2)
        test_rmse2.append(test_batch_rmse2)
        test_test_output.append(test_batch_test_output)
    time_predict_end = time.time()
    test_test_output = np.array(test_test_output)
    loss2 = sum(test_loss2[0:testtotalbatch])
    rmse2 = sum(test_rmse2[0:testtotalbatch])


    test_output = np.reshape(test_test_output, [-1, num_nodes])

    testY = testY[0:int(test_output.shape[0] / pre_len), :, :]

    test_label = np.reshape(testY, [-1, num_nodes])

    Length=int(test_output.shape[0] / pre_len)

    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)

    # if data_name =="chengdu":
    #     rmse, mae, acc, r2_score, var_score = evaluation(test_label[test_label.shape[0]-139: test_label.shape[0]-1], test_output[test_label.shape[0]-139: test_label.shape[0]-1])
    #
    # if data_name == "xian":
    #     rmse, mae, acc, r2_score, var_score = evaluation(test_label[test_label.shape[0]-416: test_label.shape[0]-1], test_output[test_label.shape[0]-416: test_label.shape[0]-1])


    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(test_rmse2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(train_batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse * max_value),
          'test_acc:{:.4}'.format(acc))

    if restore == 'yes':
        if epoch == 1:
            if method == 'dynamic':
                print('out/sdgcn/sdgcn_' + data_name + '_lr' + str(lr) +'_batch32_unit' + str(gru_units) +'_seq12_pre' + str(pre_len) + '_epoch' + restore_epoch + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
                saver.restore(sess, 'out/sdgcn/sdgcn_' + data_name + '_lr' + str(lr) +'_batch32_unit' + str(gru_units) + '_seq12_pre' + str(pre_len) + '_epoch' + restore_epoch + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
            elif method == 'fix':
              print('out/sdgcn/sdgcn_' + data_name +'_lr1e-05_batch32_unit256_seq12_pre' + str(pre_len) + '_epoch' + restore_epoch + '/fix_' + data_name + '_pre' + str(pre_len)  + '/' + data_name + '_fix_pre' + str(pre_len))
              saver.restore(sess, 'out/sdgcn/sdgcn_' + data_name +'_lr1e-05_batch32_unit256_seq12_pre'+ str(pre_len) + '_epoch' + restore_epoch + '/fix_' + data_name + '_pre' + str(pre_len)  + '/' + data_name + '_fix_pre' + str(pre_len))

    elif restore == 'no':
        if epoch == (training_epoch - 1) :
            if method == 'dynamic':
                saver.save(sess, path + '/dynamic_' + data_name + '_sparseHMM_pre' + str(pre_len) + '/' + data_name + '_sparseHMM_pre' + str(pre_len))
            elif method == 'fix':
                print(path + '/fix' + data_name + '_pre' + str(pre_len) + '/' + data_name + '_fix_pre' + str(pre_len))
                saver.save(sess, path + '/fix_' + data_name + '_pre' + str(pre_len) + '/' + data_name + '_fix_pre' + str(pre_len))

time_end = time.time()
if show_time == 'yes':
    print('Total training time', time_end - time_start, 's')
    print('Each training time', time_trainning_end - time_trainning_start, 's')
    print('Each predicting time', time_predict_end - time_predict_start, 's')

############## visualization ###############
b = int(len(train_batch_rmse) / traintotalbatch)
train_batch_rmse1 = [i for i in train_batch_rmse]
train_rmse = [(sum(train_batch_rmse1[i * traintotalbatch:(i + 1) * traintotalbatch]) / traintotalbatch) for i in
              range(b)]
train_batch_loss1 = [i for i in train_batch_loss]
train_loss = [(sum(train_batch_loss1[i * traintotalbatch:(i + 1) * traintotalbatch]) / traintotalbatch) for i in
              range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path + '/test_result.csv', index=False, header=False)

np.savetxt(path +"ground_truth.csv", test_label1, delimiter=',')




if vision == "True":
    plot_result(test_result, test_label1, path, num_nodes,data_name)
    plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)

print('Total training time', time_end - time_start, 's')
print('Each training time', time_trainning_end - time_trainning_start, 's')
print('Each predicting time', time_predict_end - time_predict_start, 's')

print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])
