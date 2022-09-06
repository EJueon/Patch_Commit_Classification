from tqdm import tqdm
import time
import pickle, gzip 
import copy
import re
import os
import pandas as pd
from transformers import AutoTokenizer
import random

pattern = ['corruption', 'crash', 'race condition', 'NULL ptr dereference', 'use-after-free', 
           'KASAN', 'caused by NULL', 'dereference', 'uninitialized variable', 'double unlink', 'double free', 'double fetch', 'memory leak', 'leaked-keyring', 'vulnerability', 'vulnerabilities', 'infoleak', 'overflow', 'underflow', 'check length passed',
           'use after free', 'check permissions', 'Kernel panic', 'information leak', 'out of bound', 'potential panic', 'non-privilege', 'unprivileged', 'cache leak',
           'operand cache leak', 'out-of-bound', 'out-of-bounds', 'fix race', 'incorrect sign extension', 'data overrun', 'reference leaks', 'leak map', 'without being initialized', 'uninitialized',
           'division by 0', 'drop reference', 'Fix race', 'sleep-while-atomic', 'syzkaller', 'sanity check','leak stack', 'OOB', 'spectre', 'fix a panic', 'kernel BUG', 'CVE-', ' race', 'permission check', 'BUG', 'infinite loop' , 'handling', 'oops']  
tags = ['commit', 'title', 'desc', 'number']
line_pattern = ['Author:', 'Date:', '1.', '2.', '3.', '[1]']
desc_pattern = ['Reported-and-tested-by:', 'Fixes:', 'Signed-off-by:', 'Reviewed-by:', 'Signed-off-by:', 
                'Tested-by:', 'Crash Report:', 'Cc:', 'Reported-by:', 'Acked-by:', 'Thanks to', 'runstate information:', 
                'WARNING', '---', '[ ', 'hex dump', '=====', 'syzbot reported:', 'Uninit was', '/*', 
                'Modules linked in:', 'this order of calls:', '#define', '#include <' '[<f', 'address:', 'ffff'
                'Call Trace:', 'task:', 'RIP:', 'RSP:', '@syzkaller:', 'usage trace:'
                ]

batch_size = 32
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def load_data(filepath):
    if not filepath.endswith('.pickle') and not filepath.endswith('.pickle.gz'):
        print("파일 형식이 올바르지 않습니다.")
        return None
    if filepath.endswith('.pickle.gz'):
        with gzip.open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        if not dataset:
            return False
        features = []
        total = len(dataset)
        progress_bar = tqdm(dataset, bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}')
        num_set = set()
        for cnt, data in enumerate(progress_bar):
            progress_bar.set_description(f'{cnt}/{total}')
            if data['number'] in num_set:
                continue
            if len(data['title'])<2 or len(data['desc'])<5:
                continue
            num_set.add(data['number'])
            title = data['title']
            desc = data['desc']
            number = data['number']
            label = data['label']
            feature = {'title': title, 'desc': desc, 'number': number, 'label': label}
            features.append(feature)
    else:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        if not dataset:
            return False
    
        features = []
        total = len(dataset)
        progress_bar = tqdm(dataset, bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}')
        for cnt, data in enumerate(progress_bar):
            progress_bar.set_description(f'{cnt}/{total}')
            
            title = data[tags[0]][tags[1]]
            desc = data[tags[0]][tags[2]]
            number = data[tags[0]][tags[3]]
            feature = {'title': title, 'desc': desc, 'number': number}
            feature = preprocess(feature)
            featrure = check_vul_label(feature)
            features.append(feature)
    return features
    
def preprocess(feature):
    desc = copy.deepcopy(feature['desc'])
    for d_p in desc_pattern:
        if len(desc) < 1:
            break
        if d_p in desc:
            desc = desc.split(d_p)[0]
    desc = re.sub(r'Link:.*\n', '', desc)
    desc = re.sub('\n', ' ', desc)
    
    feature['desc'] = desc 
    return feature

def check_vul_label(feature):
    feature['label'] = 0
    for p in pattern:
        if p in feature['title']:
            feature['label'] = 1
            break
        elif p in feature['desc']:
            feature['label'] = 1
            break
    return feature
            
def batch_iterator(data):
    for i in range(0, len(data)):
        feature = ' '.join(data[i]['title'])
        yield data[i]['title']

def vocab_hugginface(data, vocab_size=1000):
    batch_size = 32
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(data), 
                                                      vocab_size=vocab_size)
    new_tokenizer.save_pretrained("vulpatch-tokenizer")
    return new_tokenizer

def load_tokenizer(tokenizer_path):
    print(tokenizer_path)
    cache_dir = '/'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer   

def save_data(data, savepath, filename):
    filepath = os.path.join(savepath, f'{filename}')
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(data, f)

def save_csv(data, savepath, filename):
    filepath = os.path.join(savepath, f'{filename}')
    csv_f = filepath + '.csv'
    try:
        df = pd.DataFrame(data)
        # nan_value = float("NaN")
        # df.replace("", nan_value, inplace=True)
        # df.dropna(inplace=True)
        print(f"Saved Data Count : {df.shape[0]}")
        df.to_csv(csv_f, header=True, index=False, sep=',')
        return True
    except Exception as e: 
        print(e)
        return False


if __name__ == '__main__':
    data = load_data('../data/_valid_data.pickle.gz')
    print(len(data))
    # save_data(data, '../data', '_syzbot_data.pickle.gz')
    # time.sleep(10)
    # data = load_data('../data/_syzbot_data.pickle.gz')
    # save_data(data, '../data', '_syzbot_data.pickle.gz') 
    # save_csv(data, '../data', '_syzbot_data')
    # random.shuffle(data)
    # f_len = len(data) // 10
    # train_data = data[:f_len * 7]
    # valid_data = data[f_len * 7 : f_len * 8]
    # test_data = data[f_len * 8 : len(data)]

    # save_data(train_data, '../data', '_train_data.pickle.gz')
    # save_data(valid_data, '../data', '_valid_data.pickle.gz')
    # save_data(test_data, '../data', '_test_data.pickle.gz')
    
    # save_csv(train_data, '../data', '_train_data')
    # save_csv(valid_data, '../data', '_valid_data')
    # save_csv(test_data, '../data', '_test_data')
    # tokenizer = vocab_hugginface(train_data)
    
    # data = load_data('../data/_train_data.pickle.gz')
    # save_csv(data, '../data', '_train_data')
