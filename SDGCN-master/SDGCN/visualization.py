#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_result(test_result,test_label1,path,num,dataset):
    ##all test result visualization

    fig1 = plt.figure(figsize=(14,4.3))

    ax1 = fig1.add_subplot()
#    ax1 = fig1.add_subplot(1,1,1)

    ax1.set_ylabel('Traffic speed(km/h)', fontsize=30)
    ax1.set_xlabel('Time', fontsize='30')
        # ax1.set_title("Percentage %",fontsize='10',loc="left")
    if dataset == "xian":
    # 西安配置
        ax1.set_xticklabels([0, "2018-11-11", " ", "2018-11-12", " ", "2018-11-13"," ", "2018-11-15"], fontsize=25)
        ax1.set_yticklabels([0, 10, '', 20, ' ', 30, ' ', 40,], fontsize=25)
        a_pred = test_result[:, 4]
        a_true = test_label1[:, 4]
    elif dataset == "chengdu":
    # 成都配置
        ax1.set_xticklabels([0, "2018-11-08", " ", "2018-11-09", " ", "2018-11-10",], fontsize=25)
        # ax1.set_yticklabels([0, 20, ' ', 30, ' ', 40], fontsize=15)
        ax1.set_yticklabels([0, 10, '', 20, ' ', 30, ' ', 40, ], fontsize=25)
        a_pred = test_result[:, 11]
        a_true = test_label1[:, 11]


    ax1.plot(a_pred,'r-',label='prediction')
    ax1.plot(a_true,'b-',label='true')
    ax1.legend(loc='lower right',fontsize=20)

    plt.show()

    fig1.savefig(dataset + '_prediction-total-day.pdf')

    ## oneday test result visualization
    fig1 = plt.figure(figsize=(14,4.3))
    ax1 = fig1.add_subplot(1, 1, 1)

    if dataset == "xian":
        a_pred = test_result[0:144, 4]
        a_true = test_label1[0:144, 4]
    elif dataset == "chengdu":
        a_pred = test_result[0:144, 11]
        a_true = test_label1[0:144, 11]
    ax1.set_ylabel('Traffic speed(km/h)', fontsize=30)
    ax1.set_xlabel('Time', fontsize='30')

    ax1.set_xticklabels([0, "0:00", "3:20", "6:40", "10:00", "13:20", "16:40", "20:00", "23:20", "0:00"], fontsize=25)
    ax1.set_yticklabels([0, 10, ' ', 20, ' ', 30, ' ', 40], fontsize=25)

    ax1.plot(a_pred,'r-',label="prediction")
    ax1.plot(a_true,'b-',label="true")
    ax1.legend(loc='lower right',fontsize=20)


    fig1.savefig(dataset + '_prediction-oneday.pdf')

    plt.show()



    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()


