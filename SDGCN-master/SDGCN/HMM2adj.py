import numpy as np
import sys
import pandas as pd

dataset = 'chengdu'

add_markov_num = 1

road_num = {}


def load_adj(dataset):
    # 从原始道路拓扑表读取道路编号，并且生成顺序的字典键值对
    with open("HMM_convert_adj_batch//" + dataset + '_topo.csv', encoding='utf-8-sig') as topology:
        node_num = len(topology.readlines())
        # print(node_num)
    with open("HMM_convert_adj_batch//" + dataset + '_topo.csv', encoding='utf-8-sig') as topology:
        for j in range(node_num):
            line = topology.readline()
            line = line.strip()
            line = line.split(',')
            road_num.update({line[0]: j})

    # for k, v in road_num.items():
    #     print(k, v)

    # 生成单位矩阵


    if dataset == "chengdu":
        time_length = 45
    elif dataset == 'xian':
        time_length = 68

    adj_matrix = np.zeros((time_length, node_num, node_num), dtype=int)

    # print('../' + dataset + '/result ' + str(0+1) + '.txt')

    # 双层马尔可夫链结果转矩阵
    for i in range(time_length):
        adj = np.identity(node_num, dtype=int)
        with open("HMM_convert_adj_batch//" + dataset + '//result ' + str(i+1) + '.txt', encoding='utf-8-sig') as HMM:
            HMM_length = len(HMM.readlines())
            # print(HMM_length)
        with open("HMM_convert_adj_batch//" + dataset + '//result ' + str(i+1) + '.txt', encoding='utf-8-sig') as HMM:
            for j in range(HMM_length):
                line = HMM.readline()
                line = line.strip()
                line = line.split(',')
                # print(road_num.get(line[0]))
                for k in range(add_markov_num):
                    if int(line[k + 1]) == 0:
                        continue
                    else:
                        adj[road_num.get(line[0]), road_num.get(line[k + 1])] = 1
        adj = (np.transpose(adj) + adj)

        for m in range(node_num):
            for n in range(node_num):
                if adj[m, n] == 2:
                    adj[m, n] = 1
        adj_matrix[i] = adj

    return adj_matrix

# np.savetxt('adj.csv', adj_matrix, delimiter=',', fmt='%d')
