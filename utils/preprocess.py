from transformers import AutoTokenizer
from tqdm import tqdm
import pickle 
import copy
import re
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
        preprocess(feature)
        check_vul_label(feature)
        features.append(feature)
    f_len = len(features) // 10
    train_features = features[:f_len * 7]
    valid_features = features[f_len * 7 + 1 : f_len * 8]
    test_features = features[f_len * 8 + 1 : len(features)]
            
    return train_features, valid_features, test_features

def preprocess(feature):
    desc = copy.deepcopy(feature['desc'])
    for d_p in desc_pattern:
        if len(desc) < 1:
            break
        if d_p in desc:
            desc = desc.split(d_p)[0]
    desc = re.sub('\n', ' ', desc)
    feature['desc'] = desc 

def check_vul_label(feature):
    feature['label'] = 0
    for p in pattern:
        if p in feature['title']:
            feature['label'] = 1
            break
        elif p in feature['desc']:
            feature['label'] = 1
            break
            
def batch_iterator(data):
    for i in range(0, len(data)):
        feature = ' '.join(data[i]['title'])
        yield data[i]['title']

def vocab_hugginface(data, vocab_size=10000):
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

if __name__ == '__main__':
    features = load_data('syzbot_data.pickle')
    tokenizer = vocab_hugginface(features)