#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 17:31
# @Author  : zhixiuma
# @File    : dataset_info.py
# @Project : FedEPoison_2
# @Software: PyCharm
import json
import numpy as np
import csv


def count(triples):

    triples = np.array(triples)
    head_entity = triples[:,0]
    tail_entity = triples[:,2]
    relation = np.unique(triples[:,1])

    entities = np.unique(list(head_entity)+list(tail_entity))
    return len(relation),len(entities),len(triples)

import os

save_path = './data_info/'
def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)

judge_dir(save_path)
list_0 = []
for dataset in ['NELL995','WNRR']:# ,
    print(dataset)
    for client in [2,3]:
        with open('./client_data_927/'+dataset+'_'+str(client)+'_927_10_10.json','r') as file:
            data = json.load(file)
            for c in range(client):
                client_c = data[c]
                list_1 = []
                for s in ['train','valid','test']:
                    triples = client_c[s]
                    num_rel,num_ent,num_tri = count(triples)
                    list_1.append(num_rel)
                    list_1.append(num_ent)
                    list_1.append(num_tri)

                list_0.append(list_1)
print(list_0)
with open(save_path+'_1.csv', mode='w',
          encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(list_0)







