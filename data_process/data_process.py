import os
import zipfile
import pandas as pd
import numpy as np
import math
import json
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse
import linecache
import torch
import random
import argparse
import logging
import time, datetime
import csv
from ast import literal_eval
#os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
#from sparse_dot_mkl import dot_product_mkl

def filter_k_core(record, k_core, filtered_column, count_column):

    stat = record[[filtered_column, count_column]] \
               .groupby(filtered_column) \
               .count() \
               .reset_index() \
               .rename(index=str, columns={count_column: 'count'})
        
    stat = stat[stat['count'] > k_core]

    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])

    return record

def reindex_core(record, column_name):

    keys = record[column_name].unique()
    num_entities = record[column_name].nunique()
    reindex_map = {keys[i]: i for i in range(num_entities)}

    record[column_name] = record[column_name].map(reindex_map)

    reindex_map = {str(keys[i]): i for i in range(num_entities)}

    return record, reindex_map

def reindex(record, column_names):

    if not isinstance(column_names, (list, tuple)):

        column_names = [column_names]
        
    reindex_maps = []
        
    for column in column_names:

        record, reindex_map = reindex_core(record, column)
        reindex_maps.append(reindex_map)
        
    if len(reindex_maps) == 1:

        reindex_maps = reindex_maps[0]
        
    return record, reindex_maps

# percentage split

def split_core(record, splits):

    train_record = record[record['rank'] > splits[2] + splits[1]]
        
    val_test_record = record[record['rank'] <= splits[2] + splits[1]].copy()
    #val_test_record['rank'] = val_test_record.groupby('user_id')['rank'].transform(np.random.permutation)

    val_record = val_test_record[val_test_record['rank'] > splits[2]]
    test_record = val_test_record[val_test_record['rank'] <= splits[2]]

    # drop_rank_and_reset_index
    train_record = train_record.drop(columns=['rank']).reset_index(drop=True)
    val_record = val_record.drop(columns=['rank']).reset_index(drop=True)
    test_record = test_record.drop(columns=['rank']).reset_index(drop=True)
    return train_record, val_record, test_record

def percentage_split(record, splits):
    record['rank'] = record['timestamp'].groupby(record['user_id']).rank(method='first', pct=True, ascending=False)
    train_record, val_record, test_record = split_core(record, splits)
    return train_record, val_record, test_record

def clean_data(behavior_df, k_core = 10, sample_frac=0.05):
    #behavior_df = behavior_df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    behavior_df = behavior_df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id']).reset_index(drop=True)
    print(len(behavior_df))

    # filter item with multiple categories
    item_cate = behavior_df[['item_id', 'category_id']].drop_duplicates().groupby('item_id').count().reset_index().rename(columns={'category_id': 'count'})
    items_single_cate = item_cate[item_cate['count'] == 1]['item_id']
    behavior_df = behavior_df.merge(items_single_cate, on='item_id')
    print(len(behavior_df))

    # sample 
    sample_col = behavior_df['user_id'].drop_duplicates().sample(frac=sample_frac)
    behavior_df_sample = behavior_df.merge(sample_col, on='user_id').reset_index(drop=True)
    #print(behavior_df_sample.head(5))
    print(len(behavior_df_sample))
    
    behavior_df_filter = filter_k_core(behavior_df_sample, k_core, 'item_id', 'user_id')
    behavior_df_10_core = filter_k_core(behavior_df_filter, k_core, 'user_id', 'item_id')
    print(len(behavior_df_10_core))

    behavior_df_10_core, user_reindex_map = reindex_core(behavior_df_10_core, 'user_id')
    behavior_df_10_core, item_reindex_map = reindex_core(behavior_df_10_core, 'item_id')
    behavior_df_10_core, cate_reindex_map = reindex(behavior_df_10_core, 'category_id')
    #print(behavior_df_10_core.head(5))
    print(len(behavior_df_10_core))
    return behavior_df_10_core

def reindex_core_cate(df, column_name):
    reindex_map = {}
    df_list = df.to_dict(orient='list')
    for i in range(len(df_list['category'])):
        cates = df_list['category'][i]
        new_cate = []
        for cate in cates:
            if cate not in reindex_map:
                reindex_map[cate] = len(reindex_map)
            new_cate.append(reindex_map[cate])
        df_list['category'][i] = tuple(new_cate)
    #print(len(reindex_map), max(reindex_map.values()))
    return pd.DataFrame(df_list), reindex_map

def clean_data_Amazon(behavior_df, k_core = 10):
    #behavior_df = behavior_df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    behavior_df = behavior_df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id']).reset_index(drop=True)
    #print(len(behavior_df))
    
    behavior_df_filter = filter_k_core(behavior_df, k_core, 'item_id', 'user_id')
    behavior_df_10_core = filter_k_core(behavior_df_filter, k_core, 'user_id', 'item_id')
    #print(len(behavior_df_10_core))

    behavior_df_10_core, user_reindex_map = reindex_core(behavior_df_10_core, 'user_id')
    behavior_df_10_core, item_reindex_map = reindex_core(behavior_df_10_core, 'item_id')
    behavior_df_10_core, cate_reindex_map = reindex_core_cate(behavior_df_10_core, 'category')
    #print(behavior_df_10_core.head(5))
    print(len(behavior_df_10_core))
    return behavior_df_10_core

def clean_data_tafeng(behavior_df, k_core = 10):
    #behavior_df = behavior_df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    behavior_df = behavior_df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id']).reset_index(drop=True)
    #print(len(behavior_df))
    
    behavior_df_filter = filter_k_core(behavior_df, k_core, 'item_id', 'user_id')
    behavior_df_10_core = filter_k_core(behavior_df_filter, k_core, 'user_id', 'item_id')
    #print(len(behavior_df_10_core))

    behavior_df_10_core, user_reindex_map = reindex_core(behavior_df_10_core, 'user_id')
    behavior_df_10_core, item_reindex_map = reindex_core(behavior_df_10_core, 'item_id')
    behavior_df_10_core, cate_reindex_map = reindex_core(behavior_df_10_core, 'category')
    #print(behavior_df_10_core.head(5))
    print(len(behavior_df_10_core))
    return behavior_df_10_core

def clean_data_MovieLens(behavior_df, k_core = 10):
    #behavior_df = behavior_df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    behavior_df = behavior_df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id']).reset_index(drop=True)
    #print(len(behavior_df))
    
    behavior_df_filter = filter_k_core(behavior_df, k_core, 'item_id', 'user_id')
    behavior_df_10_core = filter_k_core(behavior_df_filter, k_core, 'user_id', 'item_id')
    #print(len(behavior_df_10_core))

    behavior_df_10_core, user_reindex_map = reindex_core(behavior_df_10_core, 'user_id')
    behavior_df_10_core, item_reindex_map = reindex_core(behavior_df_10_core, 'item_id')
    #print(behavior_df_10_core.head(5))
    print(len(behavior_df_10_core))
    return behavior_df_10_core

def run_ml_1m(DATASET):
    RAW_PATH = os.path.join('../datasets/', DATASET)
    if not os.path.exists(os.path.join('../datasets/', DATASET+'_my')):
        os.makedirs(os.path.join('../datasets/', DATASET+'_my'))

    ratings_name=['user_id','item_id','ratings','timestamp']
    users_name=['user_id','gender','age','occupation','zip']
    movies_name=['item_id','title','category']
    category_d = {}

    ratings_df = pd.read_table(os.path.join(RAW_PATH, 'ratings.dat'), sep='::', header=None, names=ratings_name, engine='python', encoding='ISO-8859-1')
    users_df = pd.read_table(os.path.join(RAW_PATH, 'users.dat'), sep='::', header=None, names=users_name, engine='python', encoding='ISO-8859-1')
    movies_df = pd.read_table(os.path.join(RAW_PATH, 'movies.dat'), sep='::', header=None, names=movies_name, engine='python', encoding='ISO-8859-1')
    logging.info(movies_df.head(5))

    for index, data in movies_df.iterrows():
        categories = data['category'].split('|')
        for cate in categories:
            if cate not in category_d:
                category_d[cate] = len(category_d)
        movies_df['category'][index] = tuple(set([category_d[val] for val in categories]))
    logging.info(movies_df.head(5))

    behavior_df = ratings_df.merge(movies_df, on='item_id')
    logging.info(len(behavior_df))

    # generate k-core dataset
    behavior_df_10_core = clean_data_MovieLens(behavior_df, k_core=10)

    logging.info('#user: {}, #movie： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_10_core['user_id'].nunique(), 
                                                                      behavior_df_10_core['item_id'].nunique(), 
                                                                      len(behavior_df_10_core), 
                                                                      len(behavior_df_10_core)/behavior_df_10_core['item_id'].nunique(), 
                                                                      len(behavior_df_10_core)/behavior_df_10_core['user_id'].nunique(), 
                                                                      (behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique()-len(behavior_df_10_core))/(behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique())))
    
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = percentage_split(behavior_df_10_core, [0.8, 0.1, 0.1])
    behavior_df_10_core_tr.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/train.txt', sep=',', index=False, header=None)
    behavior_df_10_core_val.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/val.txt', sep=',', index=False, header=None)
    behavior_df_10_core_ts.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/test.txt', sep=',', index=False, header=None)
    behavior_df_10_core.iloc[:,[1,5]].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_category.txt', sep=',', index=False, header=None)
    behavior_df_10_core.iloc[:,[1,4,5]].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_title_category.txt', sep='\t', index=False, header=None)
    with open('../datasets/'+DATASET+'_my/category_id.txt', 'w') as f:
        json.dump(category_d, f)
    #pd.DataFrame(category_d, index=[0]).T.to_csv('../datasets/Amazon/Books_my/category_id.txt', sep=',', header=None)

def run_ml_10M(DATASET):
    RAW_PATH = os.path.join('../datasets/', DATASET)
    if not os.path.exists(os.path.join('../datasets/', DATASET+'_my')):
        os.makedirs(os.path.join('../datasets/', DATASET+'_my'))

    ratings_name=['user_id','item_id','ratings','timestamp']
    movies_name=['item_id','title','category']
    tags_name = ['user_id', 'item_id', 'tag', 'timestamp']
    category_d = {'Adventure': 0,'Animation': 1,'Children': 2,'Comedy': 3,'Fantasy': 4,'Romance': 5,'Drama': 6,'Action': 7,'Crime': 8,'Thriller': 9, 'Horror': 10,'Mystery': 11,
    'Sci-Fi': 12, 'Documentary': 13, 'War': 14, 'Musical': 15, 'Film-Noir': 16, 'Western': 17}

    ratings_df = pd.read_table(os.path.join(RAW_PATH, 'ratings.dat'), sep='::', header=None, names=ratings_name, engine='python')
    tags_df = pd.read_table(os.path.join(RAW_PATH, 'tags.dat'), sep='::', header=None, names=tags_name, engine='python')
    movies_df = pd.read_table(os.path.join(RAW_PATH, 'movies.dat'), sep='::', header=None, names=movies_name, engine='python')
    logging.info(movies_df.head(5))

    # remove category not in the pre-defined 18 type
    remain_row = []
    for index, data in movies_df.iterrows():
        categories = data['category'].split('|')
        categories_new = [category_d[cate] for cate in categories if cate in category_d]
        if categories_new != []:
            movies_df['category'][index] = tuple(set(categories_new))
            remain_row.append(True)
        else:
            remain_row.append(False)
    logging.info(movies_df.head(5))

    movies_df.insert(1, "remain_row", remain_row)
    movies_df = movies_df[movies_df["remain_row"]==True][["item_id", "title", "category"]]

    behavior_df = ratings_df.merge(movies_df, on='item_id')
    logging.info(len(behavior_df))

    # generate k-core dataset
    behavior_df_10_core = clean_data_MovieLens(behavior_df, k_core=10)
    logging.info('#user: {}, #movie： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_10_core['user_id'].nunique(), 
                                                                      behavior_df_10_core['item_id'].nunique(), 
                                                                      len(behavior_df_10_core), 
                                                                      len(behavior_df_10_core)/behavior_df_10_core['item_id'].nunique(), 
                                                                      len(behavior_df_10_core)/behavior_df_10_core['user_id'].nunique(), 
                                                                      (behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique()-len(behavior_df_10_core))/(behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique())))

    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = percentage_split(behavior_df_10_core, [0.8, 0.1, 0.1])
    behavior_df_10_core_tr.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/train.txt', sep=',', index=False, header=None)
    behavior_df_10_core_val.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/val.txt', sep=',', index=False, header=None)
    behavior_df_10_core_ts.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/test.txt', sep=',', index=False, header=None)
    behavior_df_10_core[['item_id', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_category.txt', sep=',', index=False, header=None)
    behavior_df_10_core[['item_id', 'title', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_title_category.txt', sep='\t', index=False, header=None)
    with open('../datasets/'+DATASET+'_my/category_id.txt', 'w') as f:
        json.dump(category_d, f)

def run_amazon(DATASET):
    RAW_PATH = os.path.join('../datasets/', DATASET)
    NEW_PATH = os.path.join('../datasets/', DATASET+'_my')
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
    
    RATING_PATH = os.path.join(RAW_PATH, 'ratings_'+DATASET+'.csv')
    META_PATH = os.path.join(RAW_PATH, 'meta_'+DATASET+'.json')

    # Reading meta data
    
    review_f = open(META_PATH, 'r', encoding='utf-8')
    #item_dict = {"item_id":[], "title":[], "related": [], 'category': []}
    item_dict = {"item_id":[], "title":[], "category":[]}
    # {'also_bought': ['B00L46558G', 'B008OBJ2IS'], 'also_viewed': ['B00C7X0T68'], 'bought_together': ['B005OT3QQ2']}
    category_d = {}
    for line in tqdm(review_f):
        content = literal_eval(line)
        categories = sum(content['categories'], [])
        if len(categories) == 0:
            continue
        for category in categories:
            if category not in category_d:
                category_d[category] = len(category_d)
        item_dict["category"].append(tuple(set([category_d[val] for val in categories])))
        item_dict["item_id"].append(content["asin"])
        title = ''
        if 'title' in content:
            title = content["title"].replace(";", " ").replace("\n", " ")
        item_dict["title"].append(title)
        #related_items = ''
        #if "related" in content:
        #    if "also_bought" in content["related"]:
        #        related_items = related_items +'also_bought_'+ '_'.join(content["related"]["also_bought"])+'-'
        #    if "also_viewed" in content["related"]:
        #        related_items = related_items +'also_viewed_'+ '_'.join(content["related"]["also_viewed"])+'-'
        #    if "bought_together" in content["related"]:
        #        related_items = related_items +'bought_together_'+ '_'.join(content["related"]["bought_together"])
        #item_dict["related"].append(related_items)

    item_df = pd.DataFrame(item_dict)
    review_f.close()
    logging.info(item_df.head(5))
    logging.info(len(item_df))
    
    name = ['user_id', 'item_id', 'ratings', 'timestamp']
    ratings_df = pd.read_csv(RATING_PATH, header=None, names=name)
    print(ratings_df.head())

    behavior_df = ratings_df.merge(item_df, on='item_id')
    logging.info(len(behavior_df))

    behavior_df_10_core = clean_data_Amazon(behavior_df, k_core=10)
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = percentage_split(behavior_df_10_core, [0.8, 0.1, 0.1])
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = remove_test_new_item(behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts)
    
    behavior_df_concat = pd.concat([behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts])

    logging.info('#user: {}, #movie： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_concat['user_id'].nunique(), 
                                                                      behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat), 
                                                                      len(behavior_df_concat)/behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat)/behavior_df_concat['user_id'].nunique(), 
                                                                      (behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique()-len(behavior_df_concat))/(behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique())))
    
    behavior_df_10_core_tr.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/train.txt', sep=',', index=False, header=None)
    behavior_df_10_core_val.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/val.txt', sep=',', index=False, header=None)
    behavior_df_10_core_ts.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/test.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_category.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'title', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_title_category.txt', sep=';', index=False, header=None)
    #with open('../datasets/'+DATASET+'_my/category_id.txt', 'w') as f:
    #    json.dump(category_d, f)

def run_tafeng(DATASET):
    RAW_PATH = os.path.join('../datasets/', DATASET)
    NEW_PATH = os.path.join('../datasets/', DATASET+'_my')
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
    
    DATA_PATH = os.path.join(RAW_PATH, 'ta_feng_all_months_merged.csv')
    date_format = "%m/%d/%Y"
    data_dict = {'timestamp': [], 'user_id': [], 'age': [], 'pin': [], 'category': [], 'item_id': [], 'amount': [], 'asset': [], 'price':[], 'ratings': []}
    key_list = ['timestamp', 'user_id', 'age', 'pin','category', 'item_id', 'amount', 'asset', 'price']

    
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        for row in tqdm(csv_reader):
            date_string = row[0]
            #date_string, user_id, age, pin, category, item_id, amount, asset, price = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
            # Create a datetime object
            date_object = datetime.datetime.strptime(date_string, date_format)
            # Convert to a timestamp
            timestamp = int(time.mktime(date_object.timetuple()))   
            data_dict['timestamp'].append(timestamp)
            for i in range(1, len(key_list)):
                data_dict[key_list[i]].append(row[i])
            data_dict['ratings'].append(0)
    
    behavior_df = pd.DataFrame(data_dict)
    logging.info(behavior_df.head(5))
    logging.info(len(behavior_df))

    behavior_df_10_core = clean_data_tafeng(behavior_df, k_core=10)
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = percentage_split(behavior_df_10_core, [0.8, 0.1, 0.1])
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = remove_test_new_item_tafeng(behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts)
    
    behavior_df_concat = pd.concat([behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts])

    logging.info('#user: {}, #movie： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_concat['user_id'].nunique(), 
                                                                      behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat), 
                                                                      len(behavior_df_concat)/behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat)/behavior_df_concat['user_id'].nunique(), 
                                                                      (behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique()-len(behavior_df_concat))/(behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique())))
    
    behavior_df_10_core_tr.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/train.txt', sep=',', index=False, header=None)
    behavior_df_10_core_val.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/val.txt', sep=',', index=False, header=None)
    behavior_df_10_core_ts.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/test.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_category.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'category', 'amount', 'asset', 'price']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_title_category.txt', sep=';', index=False, header=None)
    behavior_df_10_core_tr[['user_id', 'age']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/user_age.txt', sep=';', index=False, header=None)
    #with open('../datasets/'+DATASET+'_my/category_id.txt', 'w') as f:
    #    json.dump(category_d, f)


def remove_test_new_item(train_record, val_record, test_record):
    val_record = val_record[val_record['item_id'].isin(train_record['item_id'])].reset_index(drop=True)
    test_record = test_record[test_record['item_id'].isin(train_record['item_id'])].reset_index(drop=True)
    
    # reindex item
    all_item_ids = pd.concat([train_record['item_id'], val_record['item_id'], test_record['item_id']]).unique()
    all_item_ids.sort()
    item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_item_ids)}
    train_record['item_id'] = train_record['item_id'].map(item_id_mapping)
    val_record['item_id'] = val_record['item_id'].map(item_id_mapping)
    test_record['item_id'] = test_record['item_id'].map(item_id_mapping)

    # reindex category
    all_categories = []
    for df in [train_record, val_record, test_record]:
        all_categories.extend(df['category'].explode().unique())
    unique_categories = sorted(set(all_categories))
    logging.info('category number: {}'.format(len(unique_categories)))
    category_mapping = {old: new for new, old in enumerate(unique_categories, start=0)}
    
    # a function for updata category tuple
    def update_category(tup):
        return tuple(category_mapping[x] for x in tup)

    train_record['category'] = train_record['category'].apply(update_category)
    val_record['category'] = val_record['category'].apply(update_category)
    test_record['category'] = test_record['category'].apply(update_category)
    return train_record, val_record, test_record, 

def remove_test_new_item_tafeng(train_record, val_record, test_record):
    val_record = val_record[val_record['item_id'].isin(train_record['item_id'])].reset_index(drop=True)
    test_record = test_record[test_record['item_id'].isin(train_record['item_id'])].reset_index(drop=True)
    
    # reindex item
    all_item_ids = pd.concat([train_record['item_id'], val_record['item_id'], test_record['item_id']]).unique()
    all_item_ids.sort()
    item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_item_ids)}
    train_record['item_id'] = train_record['item_id'].map(item_id_mapping)
    val_record['item_id'] = val_record['item_id'].map(item_id_mapping)
    test_record['item_id'] = test_record['item_id'].map(item_id_mapping)

    # reindex category
    all_category_ids = pd.concat([train_record['category'], val_record['category'], test_record['category']]).unique()
    logging.info('category number: {}'.format(len(all_category_ids)))
    all_category_ids.sort()
    cate_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_category_ids)}
    train_record['category'] = train_record['category'].map(cate_id_mapping)
    val_record['category'] = val_record['category'].map(cate_id_mapping)
    test_record['category'] = test_record['category'].map(cate_id_mapping)

    return train_record, val_record, test_record, 

def run_yelp(DATASET):
    RAW_PATH = "../datasets/"+DATASET+"/yelp_academic_dataset_"
    NEW_PATH = os.path.join('../datasets/', DATASET+'_my')
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
    
    BUSINESS_PATH = RAW_PATH+'business.json'
    REVIEW_PATH = RAW_PATH+'review.json'
    USER_PATH = RAW_PATH+'user.json'
    business_df = read_json(BUSINESS_PATH, ['business_id', 'name', 'address', 'city', 'state', 'stars', 'categories']).rename(columns={'business_id': 'item_id', 'name': 'title'})
    review_df = read_json(REVIEW_PATH, ['user_id', 'business_id', 'stars', 'date']).rename(columns={'business_id': 'item_id', 'stars':'ratings', 'date':'timestamp'})
    #review_df['timestamp'] = pd.to_datetime(review_df['timestamp'])
    #review_df = review_df[review_df['timestamp']>='2017-01-01']
    unique_users = review_df['user_id'].nunique()
    n = int(unique_users // 2)
    if unique_users < n:
        raise ValueError(f"只有{unique_users}个独立的user_id，无法选择{n}个。")

    selected_user_ids = review_df['user_id'].drop_duplicates().sample(n, replace=False)

    review_df = review_df[review_df['user_id'].isin(selected_user_ids)]
    review_df['timestamp'] = pd.to_datetime(review_df['timestamp'])
    user_df = read_json(USER_PATH, ['user_id', 'name'])

    category_d = {}
    category_list = []
    keep_column = []
    for index, row in tqdm(business_df.iterrows()):
        if business_df.loc[index, "categories"] is None:
            keep_column.append(False)
            continue
        keep_column.append(True)
        categories = business_df.loc[index, "categories"].split(', ')
        if len(categories) == 0:
            continue
        for category in categories:
            if category not in category_d:
                category_d[category] = len(category_d)
        category_list.append(tuple(set([category_d[val] for val in categories])))
        #business_df.loc[index, "category"] = tuple(set([category_d[val] for val in categories]))
        #business_df.loc[index, "title"] = business_df.loc[index, "title"].replace(";", " ").replace("\n", " ")
        #business_df.loc[index, "address"] = business_df.loc[index, "address"].replace(";", " ").replace("\n", " ")
        #business_df.loc[index, "city"] = business_df.loc[index, "city"].replace(";", " ").replace("\n", " ")
        #business_df.loc[index, "state"] = business_df.loc[index, "state"].replace(";", " ").replace("\n", " ")
    business_df.insert(1, "keep_column", keep_column)
    business_df = business_df[business_df['keep_column'] == True]
    business_df.insert(1, "category", category_list)
    business_df = business_df[['item_id', 'title', 'address', 'city', 'state', 'stars', 'category']]
    
    behavior_df = review_df.merge(business_df, on='item_id')
    logging.info(len(behavior_df))

    behavior_df_10_core = clean_data_Amazon(behavior_df, k_core=10)
    #logging.info('#user: {}, #item： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_10_core['user_id'].nunique(), 
    #                                                                  behavior_df_10_core['item_id'].nunique(), 
    #                                                                  len(behavior_df_10_core), 
    #                                                                  len(behavior_df_10_core)/behavior_df_10_core['item_id'].nunique(), 
    #                                                                  len(behavior_df_10_core)/behavior_df_10_core['user_id'].nunique(), 
    #                                                                  (behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique()-len(behavior_df_10_core))/(behavior_df_10_core['user_id'].nunique()*behavior_df_10_core['user_id'].nunique())))

    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = percentage_split(behavior_df_10_core, [0.8, 0.1, 0.1])
    behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts = remove_test_new_item(behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts)
    
    behavior_df_concat = pd.concat([behavior_df_10_core_tr, behavior_df_10_core_val, behavior_df_10_core_ts])

    logging.info('#user: {}, #movie： {}, #interaction: {}, #avg interaction / item: {}, #avg interaction / user: {}, sparsity: {}'.format(behavior_df_concat['user_id'].nunique(), 
                                                                      behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat), 
                                                                      len(behavior_df_concat)/behavior_df_concat['item_id'].nunique(), 
                                                                      len(behavior_df_concat)/behavior_df_concat['user_id'].nunique(), 
                                                                      (behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique()-len(behavior_df_concat))/(behavior_df_concat['user_id'].nunique()*behavior_df_concat['user_id'].nunique())))
    
    behavior_df_10_core_tr.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/train.txt', sep=',', index=False, header=None)
    behavior_df_10_core_val.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/val.txt', sep=',', index=False, header=None)
    behavior_df_10_core_ts.sort_values('timestamp')[['user_id', 'item_id', 'ratings', 'timestamp']].to_csv('../datasets/'+DATASET+'_my/test.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'category']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_category.txt', sep=',', index=False, header=None)
    behavior_df_concat[['item_id', 'title', 'category', 'address', 'city', 'state', 'stars']].drop_duplicates().to_csv('../datasets/'+DATASET+'_my/item_title_category.txt', sep='¥', index=False, header=None)
    #with open('../datasets/'+DATASET+'_my/category_id.txt', 'w') as f:
    #    json.dump(category_d, f)


def read_json(inpath, cols):
    data = []
    file = open(inpath, 'r')
    for line in tqdm(file):
        data.append(json.loads(line))
    df = pd.DataFrame(data)
    file.close()
    return df[cols]

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'ml-1m', type = str,
                        help = 'Dataset to use')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./' + args.dataset + '.log')
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    if args.dataset == "ml-1m":
        run_ml_1m(args.dataset)
    elif args.dataset == 'ml-10M':
        run_ml_10M(args.dataset)
    elif args.dataset in ['Beauty', 'Electronics', 'Video_Games', 'Books', 'Cell_Phones_and_Accessories', 'Musical_Instruments', 'Pet_Supplies', 'Office_Products']:
        run_amazon(args.dataset)
    elif args.dataset == 'Yelp2018':
        run_yelp(args.dataset)
    elif args.dataset == 'TaFeng':
        run_tafeng(args.dataset)
        # read data_process_old/gen_train_data.ipynb
        #pass
