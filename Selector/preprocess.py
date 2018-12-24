# python preprocess.py --inp_dir /raid/ltj/MRC/MSMARCO --out_dir /raid/ltj/MRC/MSMARCO/preprocessed

import os
import re
import json
import string
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from itertools import groupby
from multiprocessing import Process, Pool

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

def get_golden_passage(args):
    get_golden_passage_split(args, 'train')
    get_golden_passage_split(args, 'dev')
    print('golden passage has processed.')

def get_golden_passage_split(args, split):
    # the output format of an example is golden_passage\tquery\tanswer
    logger.info('loading preprocessed file...')
    inp_file = os.path.join(args.inp_dir, split + '_saved_in_line.txt')
    out_file = os.path.join(args.out_dir, split + '_golden_passage_with_query_answer_idx_flag_full.txt')
    to_save = []
    with open(inp_file, 'r') as f:
        exs = f.readlines()
        for ex in tqdm(exs, ascii = True, desc = 'progress report:'):
            passages = ex.split('\t')[0]
            query = ex.split('\t')[1]
            answers = ex.split('\t')[2]
            query_id = ex.split('\t')[-4]
            wellFormedAnswers = ex.split('\t')[-5]
            flag = 0 # flag == 0 means the answer in train/dev sets are standard answers.
            _answers = answers.split('#@#')
            _passages = passages.split('#@#')
            random.shuffle(_passages)
            random.shuffle(_answers)
            for psg in _passages:
                parts = psg.split('@@')
                is_selected = int(parts[0])
                content = parts[1]
                if is_selected == 1:
                    for anw in _answers:
                        if anw.strip():
                            to_save.append('\t'.join((content, query, anw, query_id, str(flag), '\n')))
            
            if wellFormedAnswers != '"[]"':
                flag = 1 # flag == 1 means the answer in train/dev sets are wellFormedAnswers.
                _answers = json.loads(wellFormedAnswers) #.split('#@#')
                random.shuffle(_passages)
                random.shuffle(_answers)
                for psg in _passages:
                    parts = psg.split('@@')
                    is_selected = int(parts[0])
                    content = parts[1]
                    if is_selected == 1:
                        for anw in _answers:
                            if anw.strip():
                                to_save.append('\t'.join((content, query, anw, query_id, str(flag), '\n')))

    with open(out_file, 'w') as f:
        f.writelines(to_save)

def get_golden_other_passage(args):
    # for each example, 
    # if it has a golden passage, select the golden passage and an other passage to get a positive example and a negative example.
    # if it has no golden passage, select an other passage to get an negative example. 
    get_golden_other_passage_split(args, 'train')
    get_golden_other_passage_split(args, 'dev')
    print('golden passage has processed.')

def get_golden_other_passage_split(args, split):
    # the output format of an example is golden|other_passage\tquery\tanswer
    logger.info('loading preprocessed file...')
    inp_file = os.path.join(args.out_dir, split + '_saved_in_line.txt')
    out_file = os.path.join(args.out_dir, split + '_golden_other_passage_with_query_answer.txt')
    to_save = []
    count_golden_passage = 0
    count_other_passage = 0
    with open(inp_file, 'r') as f:
        exs = f.readlines()
        for ex in tqdm(exs, ascii = True, desc = 'progress report:'):
            passages = ex.split('\t')[0]
            query = ex.split('\t')[1]
            answers = ex.split('\t')[2]
            _passages = passages.split('#@#')
            _answers = answers.split('#@#')
            random.shuffle(_passages)
            random.shuffle(_answers)
            for psg in _passages:
                parts = psg.split('@@')
                is_selected = int(parts[0])
                content = parts[1]
                if is_selected == 1:
                    for anw in _answers:
                        if anw.strip():
                            count_golden_passage += 1
                            to_save.append('\t'.join((content, query, anw, '\n')))
            for psg in _passages:
                parts = psg.split('@@')
                is_selected = int(parts[0])
                content = parts[1]
                if is_selected == 0:
                    anw = 'No Answer Present.'
                    count_other_passage += 1
                    to_save.append('\t'.join((content, query, anw, '\n')))
                    break
    with open(out_file, 'w') as f:
        f.writelines(to_save)
    print('In '+split+' dataset, the count_golden_passage is %d, and the count_other_passage is %d', count_golden_passage, count_other_passage)

def format_dev(args, split):
    # get the evaluation format.
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.inp_dir, 'predictions.txt.golden.trAll2dev16K.self_attn_2.reduce_dim_256.only_wellFormed.eval_format.only_wellFormed_ref')
    out_file = os.path.join(args.out_dir, split + '_reference_intermidiate.txt')
    to_save = []
    with open(inp_file, 'r') as f:
        for l in f:
            answer_dict = {}
            content = json.loads(l)
            # idxs = content['passages'].keys()
            # for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):    

            answers = content['answers']# [idx]
            answer_dict['query_id'] = content['query_id'] #[idx])
            answer_dict['answers'] = answers

            to_save.append(json.dumps(answer_dict)+'\n')
    with open(out_file, 'w') as f:
        f.writelines(to_save)

def format_dev_1(args, split):
    # get a inference format。
    # each sample consists of 10 paragraphs followed by questions and answers. 
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.inp_dir, split + '_v2.1.json')
    out_file = os.path.join(args.out_dir, split + '_inference_passage10_query_answer_2.txt')
    to_save = []
    with open(inp_file, 'r') as f:
        content = json.load(f)
        idxs = content['passages'].keys()
        for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):    
            psg = ''
            passages = content['passages'][idx][:10]
            if len(passages) < 10:
                diff = 10 - len(passages)
                while diff>0:
                    passages.append({'passage_text':'apple', 'is_selected':0})
                    diff -= 1
            assert len(passages) == 10
            for passage in passages:
                psg += passage['passage_text']
                psg += '#@#'
            
            # ans = ''
            # answers = content['answers'][idx]
            # for answer in answers:
            #     ans += answer.replace('\n', '')
            #     ans += '#@#'

            # to_save.append('\t'.join([psg[:-3], content['query'][idx], ans[:-3], str(content['query_id'][idx]), '\n']))
            to_save.append('\t'.join([psg[:-3], content['query'][idx], str(content['query_id'][idx]), '\n']))
    with open(out_file, 'w') as f:
        f.writelines(to_save)

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))

def rate_rougeL_of_passage(passage, answer):
    if passage == "unknown":
        return "__NA__", -1, -1

    if normalize_answer(answer) == "yes":
        return -1, '__YES__'
    if normalize_answer(answer) == "no":
        return -1, '__NO__'

    passage_ls = len_preserved_normalize_answer(passage).split()
    answer_ls = len_preserved_normalize_answer(answer).split()

    max_rougeL = 0.0
    fake_answer = 'fake_answer'
    beta = 1.2
    for i in range(len(answer_ls)-1):
        for j in range(i+1, len(answer_ls)):
            _answer_ls = answer_ls[i:j]
            lcs = my_lcs(_answer_ls, passage_ls)
            prec = lcs/float(len(answer_ls))
            rec = lcs/float(len(_answer_ls))
            if prec == 0 or rec == 0:
                continue
            rougeL = ((1 + beta**2)*prec*rec)/float(rec + beta**2*prec)
            if rougeL > max_rougeL:
                max_rougeL = rougeL
                fake_answer = ' '.join(_answer_ls)

    return max_rougeL, fake_answer

def process_sample(sample):
    # get rougeL of each passage.
    query, answer, passages = sample
    ret = {}
    ret['query'] = query
    ret['answer'] = answer
    ret['passages'] = []
    for passage in passages:
        passage_text = passage['passage_text']
        passage_label = passage['is_selected']
        rougel_score, fake_answer = rate_rougeL_of_passage(passage_text, answer)
        psg = '#@#'.join([passage_text, str(passage_label), fake_answer, str(rougel_score)])
        ret['passages'].append(psg)
    return json.dumps(ret)

def get_rougeL_multiprocess(args, split):
    # get rougeL of each passage.
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.inp_dir, split + '_v2.1.json')
    out_file = os.path.join(args.out_dir, split + '_passages_rougeL_query_answer_standard.txt')
    pool = Pool(processes=16)
    to_save = []
    with open(inp_file, 'r') as f:
        content = json.load(f)
        idxs = content['passages'].keys()
        idxs = [idx for idx in idxs]
        for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):  
            query = content['query'][idx]
            # answer = content['wellFormedAnswers'][idx]
            # if answer == '[]':
            #     continue
            # answer = answer[0]
            answer = content['answers'][idx][0]
            passages = content['passages'][idx]
            sample = (query, answer, passages)
            to_save.append(pool.apply_async(process_sample, (sample,)))
        pool.close()
        pool.join()
    to_save = [item.get()+'\n' for item in to_save]    
    with open(out_file, 'w') as f:
        f.writelines(to_save)

def get_nagetive_sample(args, split):
    # get rougeL of each passage.
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.out_dir, split + '_passages_rougeL_query_answer_standard.txt')
    out_file = os.path.join(args.out_dir, split + '_nagetive_sample_only_from_standard_sample.txt')
    to_save = []
    with open(inp_file) as f:
        samples = f.readlines()
        for sample in tqdm(samples, ascii = True, desc = 'progress report:'):
            sample = json.loads(sample)
            query = sample['query']
            answer = sample['answer']
            passages = sample['passages']
            # if answer == 'No Answer Present.':
            #     continue
            min_score = 1.
            candidate_passage = ''
            random.shuffle(passages)
            for passage in passages:
                passage, is_selected, fake_answer, rouge_score = passage.split('#@#')
                rouge_score, is_selected = float(rouge_score), int(is_selected)
                if is_selected == 0 and rouge_score < min_score:
                    candidate_passage = passage
                    min_score = rouge_score
            to_save.append('\t'.join([candidate_passage, query, 'No Answer Present.', '\n']))
    with open(out_file, 'w') as f:
        f.writelines(to_save)

def statistic_yes_no(args, split):
    logger.info('loading ' + split + 'file...')
    inp_file = os.path.join(args.inp_dir, split + '_v2.1.json')
    total_num = 0
    start_dict = {}   
    with open(inp_file, 'r') as f:
        content = json.load(f)
    idxs = content['passages'].keys()
    for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):  
        query = content['query'][idx]
        answers = content['answers'][idx]
        passages = content['passages'][idx]
        for answer in answers:
            if normalize_answer(answer) == 'yes' or normalize_answer(answer) == 'no':
                total_num += 1
                start_dict[query.split()[0].lower()] = start_dict.get(query.split()[0].lower(), 0) + 1
                break
    _start_dict = {}
    for key in start_dict.keys():
        if start_dict[key] > 100:
            _start_dict[key] = start_dict[key]
    print('_start_dict is', _start_dict)
    print('total_num is', total_num)
    start_words = [word for word in _start_dict.keys()].append('would')
    start_with_start_words = 0
    start_with_start_words_yes_no = 0
    for idx in tqdm(idxs, ascii = True, desc = 'progress report:'):  
        query = content['query'][idx]
        answers = content['answers'][idx]
        passages = content['passages'][idx]
        start_word = query.split()[0].lower()
        if start_word in start_words:
            start_with_start_words += 1
            for answer in answers:
                if normalize_answer(answer).startswith('yes') or normalize_answer(answer).startswith('no'):
                    start_with_start_words_yes_no += 1
                    break
    print('start_with_start_words is', start_with_start_words)
    print('start_with_start_words_yes_no is', start_with_start_words_yes_no)

def make_final_dataset(args, split='train'):
    # get rougeL of each passage.
    logger.info('loading ' + split + 'file...')
    inp_file_1 = os.path.join(args.out_dir, split + '_golden_passage_with_query_answer_idx_flag_full.txt')
    inp_file_2 = os.path.join(args.out_dir, split + '_nagetive_sample_only_from_standard_sample.txt')
    inp_file_3 = os.path.join(args.out_dir, split + '_nagetive_sample_only_from_wellFormed_sample.txt')
    out_file = os.path.join(args.out_dir, split + '_golden7_other3_passage_with_query_answer_standard_wellFormed.txt')
    lines_1 = open(inp_file_1, 'r').readlines()
    lines_2 = open(inp_file_2, 'r').readlines()
    lines_3 = open(inp_file_3, 'r').readlines()
    random.shuffle(lines_1)
    random.shuffle(lines_2)
    random.shuffle(lines_3)
    lines_2 = lines_2[:237857]
    lines_3 = lines_3[:72628]
    fake_id_flag_2 = ['666666', '0', '\n']
    fake_id_flag_3 = ['666666', '1', '\n']
    for i in range(len(lines_2)):
        t = lines_2[i].split('\t')[:-1]
        t.extend(fake_id_flag_2)
        lines_2[i] = '\t'.join(t)
    for i in range(len(lines_3)):
        t = lines_3[i].split('\t')[:-1]
        t.extend(fake_id_flag_3)
        lines_3[i] = '\t'.join(t)
    lines_1.extend(lines_2)
    lines_1.extend(lines_3)
    with open(out_file, 'w') as f:
        f.writelines(lines_1)   

def main():
    # format_file(args)
    # count_is_selected(args)
    # get_correct_answer(args)
    # count_length_distribution()
    # get_golden_passage(args)
    # get_golden_other_passage(args)
    # format_dev(args, 'dev')
    # format_dev_1(args, 'dev')
    # get_rougeL(args, 'dev')
    get_rougeL_multiprocess(args, 'dev')
    # get_nagetive_sample(args, 'train')
    # statistic_yes_no(args, 'train')
    # make_final_dataset(args)


if __name__ == '__main__':
    main()

