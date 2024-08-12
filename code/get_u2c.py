import pickle
from tqdm import tqdm
import torch
import numpy as np
import os
import argparse

def read_category(path):
    cate_num = 0
    i2c = {}
    category_d = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(',')
            line = [val.strip('"(').strip(')"').strip() for val in line if val.strip('"(').strip(')"').strip() != '']
            item = int(line[0])
            category = [int(line[index]) for index in range(1, len(line))]
            for c in category:
                if c not in category_d:
                    cate_num += 1
                    category_d[c] = 1
                else:
                    category_d[c] += 1
            i2c[item] = category
    return i2c, cate_num

def read_train_data(path, i2c, cate_num, save_path):
    u2c = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(',')
            user = int(line[0])
            item = int(line[1])
            if user not in u2c:
                u2c[user] = np.zeros(cate_num)
            for item_cate in i2c[item]:
                u2c[user][item_cate] = 1
    uid_list = list(u2c.keys())
    uid_list.sort()
    result = []
    for uid in uid_list:
        result.append(u2c[uid])
    np.savetxt(save_path, np.stack(result, axis=0), fmt='%d', delimiter=',')
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'ml-1m_my', type = str,
                        help = 'Dataset to use')
    args = parser.parse_args()

    train_path = '../datasets/' + args.dataset + '/train.txt'
    category_path = '../datasets/' + args.dataset + '/item_category.txt'
    save_path = '../datasets/' + args.dataset + '/user2category.txt'
    i2c, cate_num = read_category(category_path)
    read_train_data(train_path, i2c, cate_num, save_path)
