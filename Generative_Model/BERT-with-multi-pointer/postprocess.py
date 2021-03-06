# python preprocess.py --inp_dir /home/ltj/MRC/MSMARCO --out_dir /home/ltj/MRC/MSMARCO/preprocessed

import os
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from itertools import groupby

logger = logging.getLogger()

parser = argparse.ArgumentParser(description = 'Preprocess script of Selector')
parser.add_argument('--inp_dir', type = str, help = 'the input dir of MSMARCO')
parser.add_argument('--out_dir', type = str, help = 'the save dir of the output')
parser.add_argument('--postfix', type = str, help = 'the postfix of the input file')
# parser.add_argument('--mix_ratio', type = float)
args = parser.parse_args()

def format_dev(args, filt_no_answer=True, mix_ratio = 1):
    # get a evaluation format。
    file_name = 'predictions.txt.' + args.postfix
    logger.info('loading' + file_name + ' ...')
    inp_file = os.path.join(args.inp_dir, file_name)
    out_file = os.path.join(args.out_dir, file_name+'.eval_format.'+str(mix_ratio))
    saved_candidate = []
    to_save = []
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    samples = [lines[i:i+10] for i in range(0, len(lines), 10)]
    for sample in samples:
    	sample_idx = sample[0].split('\t')[0]
    	max_score = 0
    	_candidate = {}
    	_candidate['query_id'] = int(sample_idx)
    	_candidate['answers'] = ['no answer present .']
    	for candidate in sample:
    		idx, prediction, ori_score, ps_score, _ = candidate.split('\t')
    		assert idx == sample_idx
    		ori_score = float(ori_score)
    		ps_score = float(ps_score)
    		score = mix_ratio * ori_score + (1 - mix_ratio) * ps_score
    		if score > max_score:
    			if filt_no_answer and prediction == 'no answer present .':
    				continue
    			_candidate['query_id'] = int(idx)
    			_candidate['answers'] = [prediction]
    			max_score = score
    	if sample_idx in saved_candidate:
    		continue
    	else:
    		saved_candidate.append(sample_idx)
	    	to_save.append(json.dumps(_candidate) + "\n")
    with open(out_file, 'w') as f:
        f.writelines(to_save)

def main():
	for i in range(0, 11):
		format_dev(args, mix_ratio=1.*i/10)

if __name__ == '__main__':
    main()



