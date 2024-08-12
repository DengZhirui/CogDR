import requests
from tqdm import tqdm
import argparse
#from requests.packages import urllib3
import multiprocessing
import math
import os
import random

def read_items(dataset, path1, path2, save_dir1, save_dir2):
    sep = ';'
    lines_l1 = []
    with open(path1, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(sep)
            lines_l1.append(line)

    lines_l2 = []
    with open(path2, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(sep)
            lines_l2.append(line)

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    save_path = '0.txt'
    read_items_multiprocess_sub_tafeng(lines_l1, lines_l2, save_dir1, save_dir2, save_path)

def read_items_multiprocess(dataset, path, save_dir1, save_dir2):

    if dataset == "ml-1m_my" or dataset == "ml-10M_my":
        sep = '\t'
    elif dataset in ['Beauty_my', "Electronics_my", "Video_Games_my", "Books_my", "Cell_Phones_and_Accessories_my", "Musical_Instruments_my", "Office_Products_my", "Pet_Supplies_my"]:
        sep = ';'
    elif dataset == "Yelp2018_my":
        sep = '¥'

    lines_l = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(sep)
            #if len(line) != 3 or line[0] == '' or line[2] == '':
            #    print(line)
            lines_l.append(line)

    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    
    task_list = split_list_n_list(lines_l, 100)
    jobs = []
    for task in task_list:
        save_path = str(len(jobs))+'.txt'
        if dataset == "ml-1m_my" or dataset == "ml-10M_my":
            p = multiprocessing.Process(target=read_items_multiprocess_sub, args=(task, save_dir1, save_dir2, save_path))
        elif dataset in ['Beauty_my', "Electronics_my", "Video_Games_my", "Books_my", "Cell_Phones_and_Accessories_my", "Musical_Instruments_my", "Office_Products_my", "Pet_Supplies_my"]:
            p = multiprocessing.Process(target=read_items_multiprocess_sub_amazon, args=(task, save_dir1, save_dir2, save_path))
        elif dataset == "Yelp2018_my":
            p = multiprocessing.Process(target=read_items_multiprocess_sub_yelp, args=(task, save_dir1, save_dir2, save_path))

        jobs.append(p)
        p.start()

def read_items_multiprocess_sub(lines, save_dir1, save_dir2, save_path):
    entities_f = open(save_dir1+save_path, 'w')
    id2item_f = open(save_dir2+save_path, 'w')
    id2item = {}
    for line in tqdm(lines):
        item_id = line[0]
        id2item[item_id] = item_id
        category_l = line[2].split(',')
        categories = [val.strip('"(').strip(')"').strip() for val in category_l if val.strip('"(').strip(')"').strip() != '']
        year = line[1][-5:-1]
        titles = line[1][:-6].split(', ')[0].split('(')
        titles = [title.strip().strip(')').strip() for title in titles]
        entities = []
        for title in titles:
            title_id, entity = search_in_kg(title)
            if entity == []:
                continue
            id2item[title_id] = item_id
            entities += entity
        entities += [[item_id, 'Y', 'Y'+str(year)]]
        for category in categories:
            entities += [[item_id, 'C', 'C'+str(category)]]
        for entity in entities:
            #print(entity)
            entities_f.write(entity[0]+','+entity[1]+','+entity[2]+'\n')
    for _id in id2item:
        id2item_f.write(_id+','+id2item[_id]+'\n')
    entities_f.close()
    id2item_f.close()

def read_items_multiprocess_sub_amazon(lines, save_dir1, save_dir2, save_path):
    entities_f = open(save_dir1+save_path, 'w')
    id2item_f = open(save_dir2+save_path, 'w')
    id2item = {}
    for line in tqdm(lines):
        item_id = line[0]
        id2item[item_id] = item_id
        category_l = line[2].split(',')
        categories = [val.strip('"(').strip(')"').strip() for val in category_l if val.strip('"(').strip(')"').strip() != '']
        #year = line[1][-5:-1]
        title = line[1].replace('&amp;', '')
        #titles = [title.strip().strip(')').strip() for title in titles]
        entities = []
        #for title in titles:
        title_id, entity = search_in_kg(title)
        if entity == []:
            continue
        id2item[title_id] = item_id
        entities += entity
        #entities += [[item_id, 'Y', 'Y'+str(year)]]
        for category in categories:
            entities += [[item_id, 'C', 'C'+str(category)]]
        for entity in entities:
            #print(entity)
            entities_f.write(entity[0]+','+entity[1]+','+entity[2]+'\n')
    for _id in id2item:
        id2item_f.write(_id+','+id2item[_id]+'\n')
    entities_f.close()
    id2item_f.close()

def read_items_multiprocess_sub_yelp(lines, save_dir1, save_dir2, save_path):
    entities_f = open(save_dir1+save_path, 'w')
    id2item_f = open(save_dir2+save_path, 'w')
    id2item = {}
    for line in tqdm(lines):
        item_id = line[0]
        id2item[item_id] = item_id
        category_l = line[2].split(',')
        categories = [val.strip('"(').strip(')"').strip() for val in category_l if val.strip('"(').strip(')"').strip() != '']
        #year = line[1][-5:-1]
        title = line[1].replace('&amp;', '')
        address = line[3].replace('&amp;', '')
        city = line[4].replace('&amp;', '')
        state = line[5].replace('&amp;', '')
        star = line[6].replace('&amp;', '')
        #titles = [title.strip().strip(')').strip() for title in titles]
        entities = []
        #for title in titles:
        title_id, entity = search_in_kg(title)
        if entity == []:
            continue
        id2item[title_id] = item_id
        entities += entity
        entities += [[item_id, 'ADD', 'ADD_'+address],[item_id,'CITY','CITY_'+city],
                     [item_id,'STATE','STATE_'+state],[item_id,'STAR','STAR'+star]]
        #entities += [[item_id, 'Y', 'Y'+str(year)]]
        for category in categories:
            entities += [[item_id, 'C', 'C'+str(category)]]
        for entity in entities:
            #print(entity)
            entities_f.write(entity[0]+','+entity[1]+','+entity[2]+'\n')
    for _id in id2item:
        id2item_f.write(_id+','+id2item[_id]+'\n')
    entities_f.close()
    id2item_f.close()

def read_items_multiprocess_sub_tafeng(lines, lines_user, save_dir1, save_dir2, save_path):
    entities_f = open(save_dir1+save_path, 'w')
    id2item_f = open(save_dir2+save_path, 'w')
    id2item = {}
    for line in tqdm(lines):
        item_id = line[0]
        id2item[item_id] = item_id
        category = line[1]
        amount, asset, price = line[2], line[3], line[4]
        entities = [[item_id, 'AMOUNT', 'AM'+amount], [item_id, 'ASSET', 'AS'+asset], [item_id, 'PRICE', 'P'+price],[item_id, 'C', 'C'+str(category)]]
        for entity in entities:
            entities_f.write(entity[0]+','+entity[1]+','+entity[2]+'\n')
    for _id in id2item:
        id2item_f.write(_id+','+id2item[_id]+'\n')
    
    for line in tqdm(lines_user):
        user_id, age = line[0], line[1]
        entities_f.write('U'+user_id+','+'AGE'+',AGE'+age+'\n')

    entities_f.close()
    id2item_f.close()

def split_list_n_list(origin_list, n):
    res_list=[]
    L=len(origin_list)
    N=int(math.ceil(L/float(n)))
    begin=0
    end=begin+N
    while begin<L:
        if end<L:
            temp_list=origin_list[begin:end]
            res_list.append(temp_list)
            begin=end
            end+=N
        else:
            temp_list=origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list

def search_in_kg(query):
    entities = []
    # Which parameters to use
    params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': query,
            'language': 'en'
        }
    
    # Fetch API
    data = fetch_wikidata(params)
    if data == 'There was an error':
        return -1, []
    data = data.json()

    # Get ID from the wbsearchentities response
    if len(data['search']) == 0:
        return -1, []
    id = data['search'][0]['id']
    # Create parameters
    params_id = {
                'action': 'wbgetentities',
                'ids':id, 
                'format': 'json',
                'languages': 'en'
            }
    
    # fetch the API
    data_id = fetch_wikidata(params_id)
    if data_id == 'There was an error':
        return -1, []
    data_id = data_id.json()

    for pid in data_id['entities'][id]['claims']:
        #for val in data_id['entities'][id]['claims'][pid]:
        #    if type(val['mainsnak']['datavalue']['value']) == dict and 'datavalue' in val['mainsnak'] and 'value' in val['mainsnak']['datavalue'] and 'id' in val['mainsnak']['datavalue']['value']:
        #    print(val)
        #    if 'id' in val['mainsnak']['datavalue']['value']:
        #        entities += [[id, pid, val['mainsnak']['datavalue']['value']['id']]]
        entities += [[id, pid, val['mainsnak']['datavalue']['value']['id']] for val in data_id['entities'][id]['claims'][pid] if 'datavalue' in val['mainsnak'] and 'value' in val['mainsnak']['datavalue'] and type(val['mainsnak']['datavalue']['value']) == dict and 'id' in val['mainsnak']['datavalue']['value']]
    return id, entities

def fetch_wikidata(params):
    #urllib3.disable_warnings()
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was an error'

def reset_index(inpath, indir1, indir2, save_path, save_path1):
    id2item = {}
    id2user = {}
    files = os.listdir(indir2)
    for file in tqdm(files):
        path = indir2 + file
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                id2item[line[0]] = 'I' + line[1]
    
    entities = []
    #entities_f = open(save_path, 'w')
    files = os.listdir(indir1)
    for file in tqdm(files):
        path = indir1 + file
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                head = line[0]
                relation = line[1]
                tail = line[2]
                if head in id2item:
                    head = id2item[head]
                if tail in id2item:
                    tail = id2item[tail]
                if relation in ['ADD', 'CITY', 'STATE', 'STAR']:
                    continue
                entities.append([head, relation, tail])
                #entities_f.write(str(head)+','+str(relation)+','+str(tail)+'\n')
    
    with open(inpath, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(',')
            user = str(line[0])
            item = str(line[1])
            entities.append(['U'+user, 'RUI', id2item[item]])
    #        entities_f.write(str(id2user[user])+','+str(ui_relation)+','+str(id2item[item])+'\n')
    #entities_f.close()
    random.seed(0)
    random.shuffle(entities)
    split_ratio = [int(len(entities)*0.8), int(len(entities)*0.9)]

    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    entities_f = open(save_path+'train.txt', 'w')
    for entity in entities[:split_ratio[0]]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()
    entities_f = open(save_path+'valid.txt', 'w')
    for entity in entities[split_ratio[0]:split_ratio[1]]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()
    entities_f = open(save_path+'test.txt', 'w')
    for entity in entities[split_ratio[1]:]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()

    #id2user_f = open(save_path1, 'w')
    #for _id in id2user:
    #    id2user_f.write(str(_id)+','+str(id2user[_id])+'\n')
    #id2user_f.close()

def reset_index_old(inpath, indir1, indir2, save_path, save_path1):
    id2item = {}
    id2user = {}
    id2relation = {}
    files = os.listdir(indir2)
    for file in tqdm(files):
        path = indir2 + file
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                id2item[line[0]] = int(line[1])
    
    entities = []
    #entities_f = open(save_path, 'w')
    files = os.listdir(indir1)
    for file in tqdm(files):
        path = indir1 + file
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                head = line[0]
                relation = line[1]
                tail = line[2]
                if head not in id2item:
                    id2item[head] = max(id2item.values())+1
                head = id2item[head]
                if relation not in id2relation:
                    id2relation[relation] = len(id2relation)
                relation = id2relation[relation]
                if tail not in id2item:
                    id2item[tail] = max(id2item.values())+1
                tail = id2item[tail]
                entities.append([head, relation, tail])
                #entities_f.write(str(head)+','+str(relation)+','+str(tail)+'\n')
    
    ui_relation = len(id2relation)
    with open(inpath, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(',')
            user = int(line[0])
            item = str(line[1])
            if user not in id2user:
                if id2user != {}:
                    id2user[user] = max(max(id2item.values()), max(id2user.values())) + 1
                else:
                    id2user[user] = max(id2item.values()) + 1
            entities.append([id2user[user], ui_relation, id2item[item]])
    #        entities_f.write(str(id2user[user])+','+str(ui_relation)+','+str(id2item[item])+'\n')
    #entities_f.close()
    random.seed(0)
    random.shuffle(entities)
    split_ratio = [int(len(entities)*0.8), int(len(entities)*0.9)]

    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    entities_f = open(save_path+'train.txt', 'w')
    for entity in entities[:split_ratio[0]]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()
    entities_f = open(save_path+'valid.txt', 'w')
    for entity in entities[split_ratio[0]:split_ratio[1]]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()
    entities_f = open(save_path+'test.txt', 'w')
    for entity in entities[split_ratio[1]:]:
        entities_f.write(str(entity[2])+'\t'+str(entity[1])+'\t'+str(entity[0])+'\n')
    entities_f.close()

    id2user_f = open(save_path1, 'w')
    for _id in id2user:
        id2user_f.write(str(_id)+','+str(id2user[_id])+'\n')
    id2user_f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'ml-1m_my', type = str,
                        help = 'Dataset to use')
    parser.add_argument('--name', default='ml-1m-', type=str, help='')
    parser.add_argument('--mode', default = '', type = str,
                        help = '')
    args = parser.parse_args()

    if args.mode == "read_item":
        if args.dataset == 'TaFeng_my':
            read_items(args.dataset, '../datasets/'+args.dataset+'/item_title_category.txt', '../datasets/'+args.dataset+'/user_age.txt', '../datasets/'+args.dataset+'/wikidata_entities/', '../datasets/'+args.dataset+'/entitiesid2item/')
        else:
            read_items_multiprocess(args.dataset, '../datasets/'+args.dataset+'/item_title_category.txt', '../datasets/'+args.dataset+'/wikidata_entities/', '../datasets/'+args.dataset+'/entitiesid2item/')
    elif args.mode == "reset_index":
        if not os.path.exists('../datasets/'+args.dataset+'/kgdata/'):
            os.makedirs('../datasets/'+args.dataset+'/kgdata/')
        reset_index('../datasets/'+args.dataset+'/train.txt', '../datasets/'+args.dataset+'/wikidata_entities/', '../datasets/'+args.dataset+'/entitiesid2item/', '../datasets/'+args.dataset+'/kgdata/'+args.name, '../datasets/'+args.dataset+'/kgid2user.txt')
        
    
