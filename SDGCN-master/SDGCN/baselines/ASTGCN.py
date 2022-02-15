# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:12:27 2019

@author: gk
"""

import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse
import numpy as np
import pandas as pd

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix

from model import ASTGCN as model





# 选择数据集为成都或者西安或者PEMS07
dataset = 'chengdu'
#learning_rate chengdu为0.004,libutil 24X4 merge=False xian为0.002，libutil 24X5 merge=False
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--max_epoch', type=int, default=80, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=12, help='Size of temporal : 6')
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate [0.97/0.92]')


parser.add_argument("--data_name", type=str, default=dataset,
                    help="the number of data documents [8/4]", required=False)
if dataset == 'chengdu':
    parser.add_argument('--num_point', type=int, default=505,
                    help='road Point Number [170/307] ', required=False)
elif dataset == 'xian':
    parser.add_argument('--num_point', type=int, default=251,
                    help='road Point Number [170/307] ', required=False)
elif dataset == 'PEMS07':
    parser.add_argument('--num_point', type=int, default=505,
                    help='road Point Number [170/307] ', required=False)
elif dataset == 'PEMS04':
    parser.add_argument('--num_point', type=int, default=307,
                    help='road Point Number [170/307] ', required=False)
elif dataset == 'PEMS08':
    parser.add_argument('--num_point', type=int, default=170,
                    help='road Point Number [170/307] ', required=False)



FLAGS = parser.parse_args()
f = FLAGS.data_name
decay=FLAGS.decay

graph_signal_matrix_filename = 'data/%s/%s.npz'%(f,f)
print(graph_signal_matrix_filename)
Length = FLAGS.length
batch_size = FLAGS.batch_size
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
optimizer = FLAGS.optimizer

if dataset == 'chengdu':
    points_per_hour = 6
elif dataset == 'xian':
    points_per_hour = 6
else:
    points_per_hour= 12

num_for_predict= 12
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2
num_of_vertices = FLAGS.num_point
if dataset == 'chengdu':
    num_of_features = 1
    merge = False
elif dataset == 'xian':
    num_of_features = 1
    merge = False
else:
    num_of_features = 3
    merge = False
model_name='ASTGCN_%s' %f
params_dir = 'experiment_A'
prediction_path = 'ASTGCN_prediction_%s' %f
wdecay=0.00


device = torch.device(FLAGS.device)

if dataset == 'chengdu':
    chengdu_adj = pd.read_csv(r'data/chengdu_adj.csv', header=None)
    adj = np.mat(chengdu_adj, dtype=float)
elif dataset == 'xian':
    xian_adj = pd.read_csv(r'data/xian_adj.csv', header=None)
    adj = np.mat(xian_adj, dtype=float)
else:
    adj_filename = 'data/' + dataset +'/distance.csv'
    # read laplace matrix
    adj = get_adjacency_matrix(adj_filename, num_nodes)


adjs=scaled_Laplacian(adj)
supports=(torch.tensor(adjs)).type(torch.float32).to(device)


print('Model is %s' % (model_name))

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))


if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    #Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = all_data['test']['target']
    # print(true_value)
    print(true_value.shape)

    # np.savetxt('a.csv', true_value[0], delimiter=',', fmt='%f')
    # training set data loader
    train_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['train']['week']),
                            torch.Tensor(all_data['train']['day']),
                            torch.Tensor(all_data['train']['recent']),
                            torch.Tensor(all_data['train']['target'])
                        ),
                        batch_size=batch_size,
                        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['val']['week']),
                            torch.Tensor(all_data['val']['day']),
                            torch.Tensor(all_data['val']['recent']),
                            torch.Tensor(all_data['val']['target'])
                        ),
                        batch_size=batch_size,
                        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
                        TensorDataset(
                            torch.Tensor(all_data['test']['week']),
                            torch.Tensor(all_data['test']['day']),
                            torch.Tensor(all_data['test']['recent']),
                            torch.Tensor(all_data['test']['target'])
                        ),
                        batch_size=batch_size,
                        shuffle=False
    )
                        
    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )
    
    # loss function MSE
    loss_function = nn.MSELoss()

    # get model's structure
    net = model(c_in=num_of_features,c_out=64,
                num_nodes=num_nodes,week=24,
                day=12,recent=24,
                K=3,Kt=3)
    net.to(device)#to cuda
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10,20], gamma=0.1, last_epoch=-1)
    
    #calculate origin loss in epoch 0
    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)
    
    # compute testing set MAE, RMSE, MAPE before training
    evaluate(net, test_loader, true_value, supports, device, epoch=0)
    
    clip = 5
    his_loss =[]
    train_time=[]
    for epoch in range(1, epochs + 1):
        train_l=[]
        start_time_train = time()
        for train_w, train_d, train_r, train_t in train_loader:
            train_w=train_w.to(device)
            train_d=train_d.to(device)
            train_r=train_r.to(device)
            train_t=train_t.to(device)
            net.train() #train pattern
            optimizer.zero_grad() #grad to 0
            
            output,_,_ = net(train_w, train_d, train_r, supports)
            loss = loss_function(output, train_t)
            #backward p
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            
            #update parameter
            optimizer.step()
            
            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l=np.mean(train_l)    
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
                  % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)
        
        # compute validation loss
        valid_loss=compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
        his_loss.append(valid_loss)
        
        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, supports, device, epoch)
        
        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                               epoch,str(round(valid_loss,2))))
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))
        
    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))
    
    
    bestid = np.argmin(his_loss)
        
    print("The valid loss on best model is epoch%s_%s"%(str(bestid+1), str(round(his_loss[bestid],4))))
    best_params_filename=os.path.join(params_path,
                                      '%s_epoch_%s_%s.params' % (model_name,
                                    str(bestid+1),str(round(his_loss[bestid],2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction,spatial_at,temporal_at = predict(net, test_loader, supports, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, supports, device, epoch)
    test_time = np.mean(end_time_test-start_time_test)
    print("Test time: %.2f" % test_time)
        
    np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            spatial_at=spatial_at,
            temporal_at=temporal_at,
            ground_truth=all_data['test']['target']
        )


    # ###### evaluation ######
    # def evaluation(a, b):
    #     rmse = math.sqrt(mean_squared_error(a, b))
    #     mae = mean_absolute_error(a, b)
    #     F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    #     r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    #     var = 1 - (np.var(a - b)) / np.var(a)
    #     return rmse, mae, 1 - F_norm, r2, var
    #
    #
    # rmse, mae, acc, r2_score, var_score = evaluation(all_data['test']['target'], prediction)
    #
    #
    
    
    
    
    
