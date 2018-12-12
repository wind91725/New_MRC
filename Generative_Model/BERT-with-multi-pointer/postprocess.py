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
args = parser.parse_args()

def format_dev(args, filt_no_answer=True):
    # get a evaluation format。
    file_name = 'predictions.txt.' + args.postfix
    logger.info('loading' + file_name + ' ...')
    inp_file = os.path.join(args.inp_dir, file_name)
    out_file = os.path.join(args.out_dir, file_name+'.eval_format')
    saved_candidate = []
    to_save = []
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    samples = [lines[i:i+10] for i in range(0, len(lines), 10)]
    for sample in samples:
    	sample_idx = sample[0].split('\t')[0]
    	max_score = -10000
    	_candidate = {}
    	_candidate['query_id'] = sample_idx
    	_candidate['answers'] = ['no answer present .']
    	for candidate in sample:
    		idx, ground_truth, prediction, score, _ = candidate.split('\t')
    		assert idx == sample_idx
    		score = float(score)
    		if score > max_score:
    			if filt_no_answer and prediction == 'no answer present .':
    				continue
    			_candidate['query_id'] = idx
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
	format_dev(args)

if __name__ == '__main__':
    main()



