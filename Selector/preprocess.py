# python preprocess.py --inp_dir /raid/ltj/MRC/MSMARCO --out_dir /raid/ltj/MRC/MSMARCO/preprocessed

import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from itertools import groupby

logger = logging.getLogger()

parser = argparse.ArgumentParser(description = 'Preprocess script of Selector')
parser.add_argument('--inp_dir', type = str, help = 'the input dir of MSMARCO')
parser.add_argument('--out_dir', type = str, help = 'the save dir of the output')
parser.add_argument('--max_len', type = int, default = 460, help = 'the max length of passages')
args = parser.parse_args()

def format_file(args):
    format_split(args, 'train')
    format_split(args, 'dev')
    print('every example in a line!')

def format_split(args, split):
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.inp_dir, split + '_v2.1.json')
    out_file = os.path.join(args.out_dir, split + '_saved_in_line.txt')
    count_dict = {}
    to_save = []
    with open(inp_file, 'r') as f:
        content = json.load(f)
        idxs = content['passages'].keys()
        for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):    
            psg = ''
            passages = content['passages'][idx]
            
            length_passages = len(passages)
            count_dict[length_passages] = count_dict.get(length_passages, 0) + 1
            
            for passage in passages:
                psg += str(passage['is_selected'])
                psg += '@@'
                psg += passage['passage_text']
                psg += '#@#'
            
            ans = ''
            # if len(content['answers'][idx]) > 1:
            #     print(content['answers'][idx])
            #     break
            answers = content['answers'][idx]
            for answer in answers:
                ans += answer.replace('\n', '')
                ans += '#@#'
            
            # wellFormedAns = ''
            # wellFormedAnswers = content['wellFormedAnswers'][idx]
            # for answer in wellFormedAnswers:
            #     wellFormedAns += answer.replace('\n', '')
            #     wellFormedAns += '#@#'

            to_save.append('\t'.join([psg[:-3], content['query'][idx], ans[:-3], json.dumps(content['wellFormedAnswers'][idx]), str(content['query_id'][idx]), content['query_type'][idx], str(idx), '\n']))
    with open(out_file, 'w') as f:
        f.writelines(to_save)
        print(count_dict)

def count_is_selected(args, split = 'train'):
    logger.info('loading preprocessed file...')
    inp_file = os.path.join(args.out_dir, split + '_saved_in_line.txt')
    logger.info('count the number of each situation')
    total_count = 0
    count_dict = {}
    with open(inp_file, 'r') as f:
        exs = f.readlines()
        for ex in tqdm(exs, ascii = True, desc = 'progress report:'):
            total_count += 1
            cur_count = 0
            passages = ex.split('\t')[0].split('#@#')
            for passage in passages:
                # print(passage)
                if int(passage[0]) == 1:
                    cur_count += 1
            count_dict[cur_count] = count_dict.get(cur_count, 0) + 1
    print('total_count is:', total_count)
    print('statiscical results:', count_dict)


def get_correct_answer(args, process_no_answer = True):
    # get_correct_answer_split(args, 'train', process_no_answer)
    get_correct_answer_split(args, 'dev', process_no_answer)
    print("selector input file in a right format!")

def get_correct_answer_split(args, split = 'train', process_no_answer = True):
    logger.info('loading preprocessed file...')
    inp_file = os.path.join(args.out_dir, split + '_saved_in_line.txt')
    out_file = os.path.join(args.out_dir, split + '_processed_selector.txt')
    logger.info('get the get answer index...')
    to_save = []
    with open(inp_file, 'r') as f:
        exs = f.readlines()
        error_ex = 0
        error_dict = {}
        for ex in tqdm(exs, ascii = True, desc = 'progress report:'):
            if split == 'train':
                target = np.array([0, ] * 10)
            else:
                target = [0, ] * 10
            query = ex.split('\t')[1]
            passages = ex.split('\t')[0]
            total_selected = 0
            _passages = passages.split('#@#')
            length_passages = len(_passages)
            
            if length_passages != 10:
                error_ex += 1
                error_dict[length_passages] = error_dict.get(length_passages, 0) + 1
            if length_passages > 10:
                _passages = _passages[:10]
            elif length_passages < 10:
                diff = 10 - length_passages
                _passages.extend(['0@@apple', ] * diff)
            
            passage_to_save = []
            for idx, passage in enumerate(_passages):
                is_selected = int(passage.split('@@')[0])
                if is_selected == 1:
                    total_selected += 1
                    target[idx] = 1
                passage_text = passage.split('@@')[1]
                passage_to_save.append(passage_text)
            
            if split == 'train':
                if total_selected > 0:
                    target = target / total_selected
                elif process_no_answer:
                    target = target + 0.1
            to_save.append('\t'.join(['@@'.join(passage_to_save), query, json.dumps(list(target)), '\n']))
    with open(out_file, 'w') as f:
        f.writelines(to_save)
    print("The passages number != 10", error_ex)
    print(error_dict)

def count_length_distribution():
    (train_queries, train_passages, train_target), (dev_queries, dev_passages, dev_target) = load_preprocessed_file()
    q_t = []
    p_t = []
    q_count_dict = {}
    p_count_dict = {}
    
    max_len = 254
    
    train_exs = len(train_queries)
    for i in range(train_exs):
        tmp_length = []
        x1 = train_queries[i]
        q_t.append(len(x1))
        for j in range(10):
            x2 = train_passages[j][i]
            # tmp_length.append(len(x1) +len(x2))
            p_t.append(len(x2))

    dev_exs = len(dev_queries)
    for i in range(dev_exs):
        tmp_length = []
        x1 = dev_queries[i]
        q_t.append(len(x1))
        for j in range(10):
            x2 = dev_passages[j][i]
            # tmp_length.append(len(x1) + len(x2))
            p_t.append(len(x2))
    # print(t)
    print(len(q_t))
    print(len(p_t))
    for item in q_t:
        _item = (item - 1) // 10
        q_count_dict[_item] = q_count_dict.get(_item, 0) + 1
    for item in p_t:
        _item = (item - 1) // 10
        p_count_dict[_item] = p_count_dict.get(_item, 0) + 1
    # for k, g in groupby(t, key=lambda x: (x-1)//10):
    #     count_dict['{}-{}'.format(k*10+1, (k+1)*10)] = len(list(g))
    # for key in q_count_dict.keys():
    #     q_count_dict[key] = q_count_dict[key] / len(q_t) * 100
    # for key in p_count_dict.keys():
    #     p_count_dict[key] = p_count_dict[key] / len(p_t) * 100

    print('q_count_dict is:', q_count_dict)
    print('p_count_dict is:', p_count_dict)

def load_preprocessed_file():
    with open(args.out_dir + '/save/train_queries.json', 'r') as fp:
        train_queries = json.load(fp)
    with open(args.out_dir + '/save/train_passages.json', 'r') as fp:
        train_passages = json.load(fp)
    with open(args.out_dir + '/save/train_target.json', 'r') as fp:
        train_target = json.load(fp)
    with open(args.out_dir + '/save/dev_queries.json', 'r') as fp:
        dev_queries = json.load(fp)
    with open(args.out_dir + '/save/dev_passages.json', 'r') as fp:
        dev_passages = json.load(fp)
    with open(args.out_dir + '/save/dev_target.json', 'r') as fp:
        dev_target = json.load(fp)
    return (train_queries, train_passages, train_target), (dev_queries, dev_passages, dev_target)



def main():
    # format_file(args)
    # count_is_selected(args)
    # get_correct_answer(args)
    # count_length_distribution()


if __name__ == '__main__':
    main()

