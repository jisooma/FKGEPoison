#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 12:18
# @Author  : zhixiuma
# @File    : generate_client_data.py
# @Project : FedE-master
# @Software: PyCharm
import os
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

sys.path.append('../')

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size

        self.hr2t = ddict(set)
        # print(self.hr2t)

        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0
        # 一个正样本，对应多个负样本
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            # 从self.hr2t[(head, relation)] 返回 negative_sample 是否在其中
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )# 不在里面，返回True
            # 筛选出来不在里面的。
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        # print('data:',data)
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, ent_mask=None):
        # print('___init_____')
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.ent_mask = ent_mask
        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        # print('collate_fn.....')
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def __getitem__(self, idx):
        # print('..___getitem.....:',idx)
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)] # tail_list
        # print(label)
        trp_label = self.get_label(label) # 三元组标签
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        # print('....get_label....')
        y = np.zeros([self.nentity], dtype=np.float32)
        # print(y.shape) # [629]
        if type(self.ent_mask) == np.ndarray:
            y[self.ent_mask] = 1.0

        for e2 in label:
            y[e2] = 1.0
        # print(y)
        return torch.FloatTensor(y)



def _load_data(file_path):
    file = open(file_path,'r')
    content = file.read()
    lines = content.split('\n')
    lines.pop()
    triples = []
    for line in lines:
        line = line.split('\t')
        triples.append(list(map(int, line)))
    return np.array(triples)

# 重新划分数据，按照关系划分。
def sample_and_save_triples_filter():
    # 函数过滤掉包含不可见实体的三元组
    def _filter_unseen_entities(x):
        ent_seen = entities
        df = pd.DataFrame(x, columns=['s', 'p', 'o'])
        filter_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
        n_removed_ents = df.shape[0] - filter_df.shape[0]
        return filter_df.values, n_removed_ents

    dic = []
    file_path = os.path.join(based_path, 'train.txt')
    train_x = _load_data(file_path)

    relations_list = train_x[:, 1]
    uni_relations_list = np.unique(relations_list)
    num_relations = len(uni_relations_list)
    average_relations = int(num_relations/client)

    for c in range(client):
        # 训练集
        # 1、从关系列表中，随机选择 average_relations个关系
        train_relation_samples = np.random.choice(uni_relations_list, average_relations, replace=False)
        # 2、根据关系列表，拿到相应的关系对应的索引
        train_relation_samples_index = [i for i, value in enumerate(relations_list) if value in train_relation_samples]
        train_triple_samples = train_x[train_relation_samples_index]

        entities = np.unique(np.array(train_triple_samples[:, [0, 2]]).flatten())

        # 验证集
        valid_file_path = os.path.join(based_path, 'valid.txt')
        valid_x = _load_data(valid_file_path)
        valid_relations_list = valid_x[:, 1]
        valid_relation_samples_index = [i for i, value in enumerate(valid_relations_list) if value in train_relation_samples]
        valid_triple_samples = train_x[valid_relation_samples_index]
        valid_x, remove = _filter_unseen_entities(valid_triple_samples)
        valid_triple_samples = valid_x

        # 测试集
        test_file_path = os.path.join(based_path, 'test.txt')
        test_x = _load_data(test_file_path)
        test_relations_list = test_x[:, 1]
        test_relation_samples_index = [i for i, value in enumerate(test_relations_list) if
                                        value in train_relation_samples]
        test_triple_samples = test_x[test_relation_samples_index]
        test_x, remove = _filter_unseen_entities(test_triple_samples)
        test_triple_samples = test_x

        dic.append({'train': train_triple_samples.tolist(), 'valid': valid_triple_samples.tolist(),
                    'test': test_triple_samples.tolist()})

    with open(save_client_path + dataset_name + '_' + str(client)+ '.json', 'w') as outfile:
        json.dump(dic, outfile)




# 重新划分数据，按照关系划分。
def sample_and_save_triples_filter_0():
    # 函数过滤掉包含不可见实体的三元组
    def _filter_unseen_entities(x):
        ent_seen = entities
        df = pd.DataFrame(x, columns=['s', 'p', 'o'])
        filter_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
        n_removed_ents = df.shape[0] - filter_df.shape[0]
        return filter_df.values, n_removed_ents

    import random

    def _select_relations(lst, average_relations):
        print(type(lst))
        print(type(average_relations))
        selected_relations = []
        for _ in range(client):  # 选取3次
            if len(lst) < average_relations:
                selected = random.sample(lst, len(lst))
                selected_relations.append(selected)
                print('break')
                # break  # 列表中的元素不足以组成一组
            else:
                selected = random.sample(lst, average_relations)
                selected_relations.append(selected)
            # 从原列表中移除已选取的元素
            for element in selected:
                lst.remove(element)
        return selected_relations

    dic = []
    file_path = os.path.join(based_path, 'train.txt')
    train_x = _load_data(file_path)

    relations_list = train_x[:, 1]
    uni_relations_list = np.unique(relations_list)
    lst = uni_relations_list.tolist()
    num_relations = len(uni_relations_list)
    average_relations = int(num_relations/client)

    selected_relations = _select_relations(lst,average_relations)

    for c in range(client):
        # 训练集
        # 根据关系列表，拿到相应的关系对应的索引
        train_relation_samples_index = [i for i, value in enumerate(relations_list) if value in selected_relations[c]]
        train_triple_samples = train_x[train_relation_samples_index]

        entities = np.unique(np.array(train_triple_samples[:, [0, 2]]).flatten())

        # 验证集
        valid_file_path = os.path.join(based_path, 'valid.txt')
        valid_x = _load_data(valid_file_path)
        valid_relations_list = valid_x[:, 1]
        valid_relation_samples_index = [i for i, value in enumerate(valid_relations_list) if value in selected_relations[c]]
        valid_triple_samples = valid_x[valid_relation_samples_index]
        valid_x, remove = _filter_unseen_entities(valid_triple_samples)
        valid_triple_samples = valid_x

        # 测试集
        test_file_path = os.path.join(based_path, 'test.txt')
        test_x = _load_data(test_file_path)
        test_relations_list = test_x[:, 1]
        test_relation_samples_index = [i for i, value in enumerate(test_relations_list) if value in selected_relations[c]]
        test_triple_samples = test_x[test_relation_samples_index]
        test_x, remove = _filter_unseen_entities(test_triple_samples)
        test_triple_samples = test_x

        dic.append({'train': train_triple_samples.tolist(), 'valid': valid_triple_samples.tolist(),
                    'test': test_triple_samples.tolist()})

    with open(save_client_path + dataset_name +"_" +str(client)+ '.json', 'w') as outfile:
        json.dump(dic, outfile)


# def sample_and_save_triples_filter_2():
#     # 函数过滤掉包含不可见实体的三元组
#     def _filter_unseen_entities(x):
#         # print(x)
#         ent_seen = entities
#         df = pd.DataFrame(x, columns=['s', 'p', 'o'])
#         filter_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
#         n_removed_ents = df.shape[0] - filter_df.shape[0]
#         return filter_df.values, n_removed_ents
#
#     dic = []
#     for c in range(client):
#         # 训练集
#         file_path = os.path.join(based_path, 'train.txt')
#         train_x = _load_data(file_path)
#
#         num_triples = len(train_x)
#         train_triples_sample_mask = np.random.choice(num_triples, int(num_train), replace=False)
#         train_triple_samples = train_x[train_triples_sample_mask]
#
#         entities = np.unique(np.array(train_triple_samples[:, [0, 2]]).flatten())
#
#         # 验证集
#         file_path = os.path.join(based_path, 'valid.txt')
#         valid_x, remove = _filter_unseen_entities(_load_data(file_path))
#         num_triples = len(valid_x)
#         if num_triples>int(num_valid):
#             valid_triples_sample_mask = np.random.choice(num_triples, int(num_valid), replace=False)
#             valid_triple_samples = valid_x[valid_triples_sample_mask]
#             print('valid_triple_samples1:',valid_triple_samples)
#         else:
#             valid_triple_samples = valid_x
#             print('valid_triple_samples2:', valid_triple_samples)
#
#         # 测试集
#         file_path = os.path.join(based_path, 'test.txt')
#         test_x, _ = _filter_unseen_entities(_load_data(file_path))
#         num_triples = len(test_x)
#         if num_triples>int(num_test):
#             test_triples_sample_mask = np.random.choice(num_triples, int(num_test), replace=False)
#             test_triple_samples = test_x[test_triples_sample_mask]
#         else:
#             test_triple_samples = test_x
#
#         dic.append({'train': train_triple_samples.tolist(), 'valid': valid_triple_samples.tolist(),
#                     'test': test_triple_samples.tolist()})
#
#     with open(save_client_path + dataset_name + '_' + str(client) + '_' + str(num_train) + '_' + str(
#             num_valid) + '_' + str(num_test) + '.json', 'w') as outfile:
#         json.dump(dic, outfile)
#
#
# def sample_and_save_triples_1():
#
#     dic = []
#     for c in range(client):
#         # 训练集
#         # print('client :',c)
#         file_path = os.path.join(based_path,'train.txt')
#         train_x = _load_data(file_path)
#         # print(train_x)
#         num_triples = len(train_x)
#         train_triples_sample_mask = np.random.choice(num_triples, int(num_train), replace=False)
#         train_triple_samples = train_x[train_triples_sample_mask]
#
#
#         # 验证集
#         file_path = os.path.join(based_path, 'valid.txt')
#         valid_x = _load_data(file_path)
#         num_triples = len(valid_x)
#         valid_triples_sample_mask = np.random.choice(num_triples,int(num_valid), replace=False)
#         valid_triple_samples = valid_x[valid_triples_sample_mask]
#
#         # 测试集
#         file_path = os.path.join(based_path, 'test.txt')
#         test_x = _load_data(file_path)
#         num_triples = len(test_x)
#         test_triples_sample_mask = np.random.choice(num_triples,int(num_test), replace=False)
#         test_triple_samples = test_x[test_triples_sample_mask]
#
#         dic.append({'train': train_triple_samples.tolist(), 'valid': valid_triple_samples.tolist(), 'test': test_triple_samples.tolist()})
#
#
#     with open(save_client_path+dataset_name+'_'+str(client) +'_'+str(num_train)+'_'+str(num_valid)+'_'+str(num_test)+ '.json', 'w') as outfile:
#         json.dump(dic, outfile)
#
# # # 划分比例0.8，0.1，0.1
# def sample_and_save_triples():
#
#     dic = []
#     for c in range(client):
#         # 训练集
#         # print('client :',c)
#         file_path = os.path.join(based_path,'train.txt')
#         train_x = _load_data(file_path)
#         # print(train_x)
#         num_triples = len(train_x)
#         train_triples_sample_mask = np.random.choice(num_triples, int(num_train), replace=False)
#         train_triple_samples = train_x[train_triples_sample_mask]
#
#
#         # 验证集
#         file_path = os.path.join(based_path, 'valid.txt')
#         valid_x = _load_data(file_path)
#         num_triples = len(valid_x)
#         valid_triples_sample_mask = np.random.choice(num_triples,int(int(num_train)/0.8*0.1), replace=False)
#         valid_triple_samples = valid_x[valid_triples_sample_mask]
#
#         # 测试集
#         file_path = os.path.join(based_path, 'test.txt')
#         test_x = _load_data(file_path)
#         num_triples = len(test_x)
#         test_triples_sample_mask = np.random.choice(num_triples,int(int(num_train)/0.8*0.1), replace=False)
#         test_triple_samples = test_x[test_triples_sample_mask]
#
#         dic.append({'train': train_triple_samples.tolist(), 'valid': valid_triple_samples.tolist(), 'test': test_triple_samples.tolist()})
#
#
#     with open(save_client_path+dataset_name+'_'+str(client) +'_'+str(num_train)+'_'+str(num_valid)+'_'+str(num_test)+ '.json', 'w') as outfile:
#         json.dump(dic, outfile)

# 由于采样,可能不包含某些实体关系,所以需要生成新的id.
def generate_new_id():

    with open(save_client_path+dataset_name + '_'+str(client) + '.json', "r") as f:
        all_data = json.load(f)
    # print('all_data:',all_data)
    # 采样的所有的实体和关系id
    all_ent = np.array([], dtype=int)
    all_rel = np.array([], dtype=int)
    for data in all_data:
        # np.union1d 返回两个数组的并集
        for st in ['train', 'valid', 'test']:
            all_ent = np.union1d(all_ent, np.array(data[st])[:, [0, 2]].reshape(-1))
            all_rel = np.union1d(all_rel, np.array(data[st])[:, [1]])

    # print(all_ent)
    nentity = len(all_ent)
    nrelations = len(all_rel)
    print('nentity:', nentity)
    print('nrelations:', nrelations)

    # 生成新的索引id: ori_id->new_id   /  new_id-->ori_id
    ent_new_ids = {}
    new_ids_ent = {}
    for idx, val in enumerate(all_ent):
        ent_new_ids[int(val)] = idx
        new_ids_ent[idx] = int(val)

    # print(ent_new_ids)
    rel_new_ids = {}
    new_ids_rel = {}
    for idx, val in enumerate(all_rel):
        rel_new_ids[int(val)] = idx
        new_ids_rel[idx] = int(val)

    # 保存新的索引id与原始索引id对应关系
    # 实体旧id映射到新的id
    with open(save_client_path+dataset_name +'_'+ str(client) +'_with_entity_new_id.json', 'w') as file1:
        json.dump(ent_new_ids, file1)
    # 关系id映射到新的id
    with open(save_client_path+dataset_name +'_'+ str(client) +'_with_relation_new_id.json', 'w') as file2:
        json.dump(rel_new_ids, file2)
    # 实体新id到旧的id
    with open(save_client_path+dataset_name +'_'+ str(client) +'_with_new_id_entity.json', 'w') as file3:
        json.dump(new_ids_ent, file3)
    # 关系新id映射到旧的id
    with open(save_client_path+dataset_name +'_'+ str(client) +'_with_new_id_relation.json', 'w') as file4:
        json.dump(new_ids_rel, file4)

    all_data_with_new_idx = []
    # 把所有客户端的实体id和关系id用新的id进行替换
    for data in all_data:
        triple_samples = {}
        for st in ['train', 'valid', 'test']:
            ent_h = np.array(data[st])[:, [0]].reshape(-1)
            ent_r = np.array(data[st])[:, [1]].reshape(-1)
            ent_t = np.array(data[st])[:, [2]].reshape(-1)
            ent_h_to_new_id = []
            r_to_new_id = []
            ent_t_to_new_id = []
            for e in ent_h:
                ent_h_to_new_id.append(ent_new_ids[int(e)])
            for e in ent_r:
                r_to_new_id.append(rel_new_ids[int(e)])
            for e in ent_t:
                ent_t_to_new_id.append(ent_new_ids[int(e)])
            # print(len(ent_h_to_new_id))
            # 三元组替换为新的id
            triple_samples[st] = np.column_stack((ent_h_to_new_id,r_to_new_id,ent_t_to_new_id)).tolist()
            # print(triple_samples[st].shape)
        all_data_with_new_idx.append(triple_samples)

    with open(save_client_path+dataset_name +'_'+ str(client) + '_with_new_id.json', 'w') as outfile:
        json.dump(all_data_with_new_idx, outfile)


def read_triples(all_data,args):
    all_ent = np.array([], dtype=int)
    all_rel = np.array([], dtype=int)

    for data in all_data:
        # np.union1d 返回两个数组的并集
        for st in ['train', 'valid', 'test']:
            all_ent = np.union1d(all_ent, np.array(data[st])[:, [0, 2]].reshape(-1))
            all_rel = np.union1d(all_rel, np.array(data[st])[:, [1]])
    nentity = len(all_ent)
    nrelation = len(all_rel)
    print('nentity:',nentity)
    print('nrelation:',nrelation)

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    rel_embed_list = []

    ent_freq_list = []
    all_train_ent_list = []
    for data in tqdm(all_data):
        train_triples = data['train']
        valid_triples = data['valid']
        test_triples = data['test']
        train_ent = []
        for st in ['train']:
            train_ent = np.union1d(all_ent, np.array(data[st])[:, [0, 2]].reshape(-1))
        print(len(train_ent))
        all_train_ent_list.append(train_ent)
        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(np.array(data['train'])[:,[0,2]].reshape(-1)), assume_unique=True)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])

        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_ent)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_ent)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_list.append(train_dataloader)

        # print(valid_triples)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )

        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.client_model in ['ComplEx']:
            rel_embed = torch.zeros(nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            rel_embed = torch.zeros(nrelation, args.hidden_dim).to(args.gpu).requires_grad_()

        nn.init.uniform_(
            tensor=rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        rel_embed_list.append(rel_embed)

        ent_freq = torch.zeros(nentity)
        for e in np.array(data['train'])[:,[0,2]].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

    ent_freq_mat = torch.stack(ent_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
           ent_freq_mat, rel_embed_list, nentity,nrelation,all_train_ent_list


if __name__=='__main__':
    from set_args import args
    save_client_path = args.save_client_path
    if not os.path.exists(save_client_path):
        os.makedirs(save_client_path)
    dataset_name = args.dataset_name
    # dataset_name = 'NELL995'
    seed = 345345
    np.random.seed(seed)
    rdm = np.random.RandomState(seed)
    rng = np.random.default_rng(seed)

    based_path = r'./{0}/'.format(dataset_name)

    client = args.num_client


    # 根据客户端需要的训练/验证/测试三元组数量,为每个客户端采样和保存相应的三元组.
    # sample_and_save_triples()
    # # 为采样的数据集,生成新的id
    # generate_new_id()
    #
    # # 按照batch,生成训练/验证/测试数据集等训练需要的数据内容
    # with open(save_client_path+dataset_name + '_'+str(client) +'_'+str(num_train)+'_'+str(num_valid)+'_'+str(num_test)+'_with_new_id.json', 'r') as outfile:
    #     all_data = json.load(outfile)
    # read_triples(all_data,args)

    # FB15k237
    sample_and_save_triples_filter_0()
    # WNRR/NELL995
    # # sample_and_save_triples()
    # # 为采样的数据集,生成新的id
    generate_new_id()
    #
    # 按照batch,生成训练/验证/测试数据集等训练需要的数据内容
    with open(save_client_path+dataset_name + '_'+str(client) +'_with_new_id.json', 'r') as outfile:
        all_data = json.load(outfile)
    read_triples(all_data,args)