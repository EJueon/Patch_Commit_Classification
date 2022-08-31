from tqdm import tqdm
import pickle, gzip
import requests, json
from bs4 import BeautifulSoup 

syzbot_base = 'https://syzkaller.appspot.com'
ERR_BC = 'Bad Commit'

def save_data(file, data):
    with gzip.open(file, 'wb') as f:
        pickle.dump(data, f)

def load_data(file):
    with gzip.open(file, 'rb') as f:
        data = pickle.load(f)
        return data

def get_diff(soup):
    tmp = soup.select('#cgit > div.content > table.diff')[0]
    tmp = str(tmp).replace('<br/>','\n')
    soup_diff = BeautifulSoup(tmp, 'html.parser')
    diff_lines = soup_diff.find_all('div')
    diff = ''
    for line in diff_lines:
        diff += line.text + '\n'
    return diff

def get_source_info(soup):
    source_info = list()
    tmp = str(soup.select('#files > div.js-diff-progressive-container')[0])
    soup_file = BeautifulSoup(tmp, 'html.parser')
    file_list = soup_file.find_all('div', class_='file')
    for file in file_list:
        file_info = dict()
        file_info['file_name'] = file.find('a').text
        file_data = BeautifulSoup(str(file), 'html.parser')
        lines = file_data.find_all('span', class_='blob-code-inner')
        before = ''
        after = ''
        for idx, line in enumerate(lines):
            if idx & 1 == 0:
                before += line.text + '\n'
            else:
                after += line.text + '\n'
        file_info['before'] = before
        file_info['after'] = after
        source_info.append(file_info)
    return source_info

def get_soup(url):
    #req = requests.get(url, timeout=10).text
    req = requests.get(url).text
    return BeautifulSoup(req, 'html.parser')

def get_commit_info(commit_short):
    soup = get_soup(f'https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit?id={commit_short}')
    try:
        tmp = soup.find_all('table', {'summary':'commit info', 'class':'commit-info'})[0]
        tmp = tmp.find_all('td', {'class':'sha1'})[0]
        tmp = tmp.find('a').get('href')
    except:
        return None
    commit = tmp.split('?id=')[-1]

    # print(f'get_info from {commit}')

    info = dict()
    info['number'] = commit
    commit_title = soup.select('#cgit > div.content > div.commit-subject')
    if len(commit_title) < 1:
        return ERR_BC
    info['title'] = commit_title[0].text.strip()
    commit_desc = soup.select('#cgit > div.content > div.commit-msg')
    info['desc'] = commit_desc[0].text.strip()
    info['diff'] = get_diff(soup)

    soup_source = get_soup(f'https://github.com/torvalds/linux/commit/{commit}?branch={commit}&diff=split')
    try:
        info['source_info'] = get_source_info(soup_source)
    except:
        return ERR_BC
    return info


def dump_all(data, level=0):
    if type(data) == list:
        for d in data:
            dump_all(d, level)
            print('')
    elif type(data) == dict:
        for k in data.keys():
            print('==='*level + f'{k} :')
            if type(data[k]) == str:
                data[k] = data[k].replace('\n','\n'+'   '*level)
                print('   '*level + f'{data[k]}\n')
            else:
                dump_all(data[k], level+1)
        print('')


def get_fixed_commit(url_post):
    soup = get_soup(syzbot_base + url_post) 
    tmp = soup.find_all('span', {'class':'mono'})[0].text
    return tmp.strip().split()[0]


def parse_onepatch(url_list):
    BC_list = list()
    data = list()
    for url in url_list:
        report = dict()
        report['report_url'] = syzbot_base + url
        report['short_commit'] = get_fixed_commit(url)
        report['commit'] = get_commit_info(report['short_commit'])
        if not report['commit']:
            continue
        if report['commit'] == ERR_BC:
            BC_list.append(report['short_commit'])
            continue
        data.append(report)
    return data
    
def parse_patch(filepath):
    version = 'linux'
    url = 'https://syzkaller.appspot.com/upstream/fixed'
    soup = get_soup(url)
    list_table = soup.find_all('table',{'class':'list_table'})[0]
    titles = list_table.find_all('td', {'class':'title'})
    
    BC_list = list()
    data = list()
    #print 'Version: %s' % version
    total = len(titles)
    cnt = 0
    # progress_bar = tqdm(titles, bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}')
    for title in titles:
        # progress_bar.set_description(f'{cnt}/{total}')
        cnt += 1
        if cnt % 10 == 0:
            print(f'{cnt}/{total}')
        report = dict()
        report['report_title'] = title.text
        report['report_url'] = syzbot_base + title.find('a').get('href')
        report['short_commit'] = get_fixed_commit(title.find('a').get('href'))
        report['commit'] = get_commit_info(report['short_commit'])
        if not report['commit']:
            continue
        if report['commit'] == ERR_BC:
            BC_list.append(report['short_commit'])
            continue
        data.append(report)

    save_data(filepath, data)
    print(f'{BC_list}')
    return filepath

if __name__ == '__main__':
    filepath = 'new_data.pickle.gz'
    parse_patch(filepath)
    #info = get_commit_info('7caac62ed598a196d6ddf8d9c121e12e082cac3a')
    #dump_all(info)
    # print("done")

