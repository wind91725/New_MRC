# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on MS_MARCO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import copy
import os
import random
import six
from tqdm import tqdm, trange
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from tokenization import load_idx_to_token
from modeling import BertConfig, BertForQuestionAnswering, BertWithMultiPointer
from optimization import BERTAdam

from parallel import DataParallelModel, DataParallelCriterion

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

P_MAX_LENGTH = 240
Q_MAX_LENGTH = 50
A_MAX_LENGTH = 100

class MarcoExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 passage_text,
                 query_text,
                 answer_text,
                 example_id=None):
        self.passage_text = passage_text
        self.query_text = query_text
        self.answer_text = answer_text
        self.example_id = example_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        if self.example_id:
            s += "qas_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", passage_text: %s" % (tokenization.printable_text(self.passage_text))
        s += ", query_text: %s" % (tokenization.printable_text(self.query_text))
        s += ", answer_text: %s" % (tokenization.printable_text(self.answer_text))
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 answer_ids=None,
                 answer_mask=None):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answer_ids = answer_ids
        self.answer_mask = answer_mask

def read_msmarco_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as f:
        data = f.readlines()
        random.shuffle(data)
        if is_training:
            data = data[:266666]
        else:
            data = data[:16666]
    
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for sample in data:
        sample = sample.split('\t') # a list of passage, query, answer and maybe an id
        tokens_lists = [[] for _ in range(len(sample))]
        for idx, item in enumerate(sample):
            prev_is_whitespace = True
            for c in item:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        tokens_lists[idx].append(c)
                    else:
                        tokens_lists[idx][-1] += c
                    prev_is_whitespace = False

        if len(tokens_lists) == 3:
            passage_text, query_text, answer_text = [' '.join(tokens_list) for tokens_list in tokens_lists]
            example = MarcoExample(                
                passage_text=passage_text,
                query_text=query_text,
                answer_text=answer_text)
        elif len(tokens_lists) == 4:
            passage_text, query_text, answer_text, example_id = [' '.join(tokens_list) for tokens_list in tokens_lists]
            example = MarcoExample(                
                passage_text=passage_text,
                query_text=query_text,
                answer_text=answer_text,
                example_id=example_id)

        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, p_max_length,
                                q_max_length, a_max_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        sample = [example.passage_text, example.query_text, example.answer_text, example.example_id]
        example_id = None
        if len(sample) == 4:
            example_id = sample[-1]
            sample = sample[:-1]
        tokens = [tokenizer.tokenize(item) for item in sample] 
        passage_tokens, query_tokens, answer_tokens = tokens
        
        passage_tokens = passage_tokens[:p_max_length]
        query_tokens = query_tokens[:q_max_length]
        answer_tokens = answer_tokens[:a_max_length]

        tokens = []
        segment_ids = []
        # Add query to the tokens
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        # Add Passage tp the tokens
        for token in passage_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        max_seq_length = p_max_length + q_max_length + 3
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        tokens = []
        # Add answer to the tokens
        tokens.append("[CLS]")
        for token in answer_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        answer_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        answer_mask = [1] * len(answer_ids)

        # Zero-pad up to the sequence length.
        max_seq_length = a_max_length + 2
        while len(answer_ids) < max_seq_length:
            answer_ids.append(0)
            answer_mask.append(0)

        assert len(answer_ids) == max_seq_length
        assert len(answer_mask) == max_seq_length

        features.append(
            InputFeatures(
                example_id=example_id,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                answer_ids=answer_ids,
                answer_mask=answer_mask))

    return features

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def eval_the_model(args, model, eval_data):
    if args.local_rank == -1:
        eval_sampler = RandomSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    eval_loss = 0
    eval_batch = 0
    n_gpu = torch.cuda.device_count()

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self

        input_ids, input_mask, segment_ids, answer_ids, answer_mask = batch
        loss, _ = model(input_ids, segment_ids, input_mask, answer_ids=answer_ids, answer_mask=answer_mask)

        eval_loss += loss.mean().detach().cpu()
        eval_batch += 1
    eval_loss /= eval_batch

    return eval_loss, math.pow(math.e, eval_loss)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--encoder_learning_rate", default=5e-5, type=float, help="The initial learning rate for Bert encoder.")
    parser.add_argument("--decoder_learning_rate", default=5e-3, type=float, help="The initial learning rate for generative decoder.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--half_dim',
                        default=False, action='store_true',
                        help='reduce the dimention of decoder.')
    parser.add_argument('--load_trained_model',
                        default=False, action='store_true',
                        help='reduce the dimention of decoder.')
    parser.add_argument('--postfix',
                        default='',
                        help='the name of log file to be saved.')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config.half_dim = args.half_dim

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    idx_to_vocab = load_idx_to_token(args.vocab_file)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        print('Preprocess the train dataset.')
        train_examples = read_msmarco_examples(
            input_file=args.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            p_max_length=P_MAX_LENGTH,
            q_max_length=Q_MAX_LENGTH,
            a_max_length=A_MAX_LENGTH)

        train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        train_answer_ids = torch.tensor([f.answer_ids for f in train_features], dtype=torch.long)
        train_answer_mask = torch.tensor([f.answer_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids,
                                   train_answer_ids, train_answer_mask)
    
    print('Preprocess the dev dataset.')
    eval_examples = read_msmarco_examples(
        input_file=args.predict_file, is_training=False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        p_max_length=P_MAX_LENGTH,
        q_max_length=Q_MAX_LENGTH,
        a_max_length=A_MAX_LENGTH)

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_answer_ids = torch.tensor([f.answer_ids for f in eval_features], dtype=torch.long)
    eval_answer_mask = torch.tensor([f.answer_mask for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_answer_ids, eval_answer_mask)

    # Prepare model
    # model = BertForQuestionAnswering(bert_config)
    model = BertWithMultiPointer(bert_config)
    if args.init_checkpoint is not None:
        print('Loading the pre-trained BERT parameters')
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        model.bert.load_state_dict(state_dict)
        decoder_embedding = ["word_embeddings.weight", "position_embeddings.weight", "token_type_embeddings.weight", "LayerNorm.gamma", "LayerNorm.beta"]
        decoder_embedding = ['embeddings.'+n for n in decoder_embedding]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in decoder_embedding:
                name = k[11:]
                new_state_dict[name] = v
        model.decoderEmbedding.load_state_dict(new_state_dict)
     
    if args.load_trained_model:
        print('Loading the pre-trained Model')
        model_path = args.output_dir+'/models/best_params.pt.'+args.postfix
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model = DataParallelModel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    # print('param_optimizer is:\n', [n for n, p in param_optimizer])
    # exit(0)
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and ('bert' in n or 'decoderEmbedding' in n)], 'weight_decay_rate': 0.01, 'lr': args.encoder_learning_rate},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and ('bert' not in n and 'decoderEmbedding' not in n)], 'weight_decay_rate': 0.01, 'lr': args.decoder_learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and ('bert' in n or 'decoderEmbedding' in n)], 'weight_decay_rate': 0.0, 'lr': args.encoder_learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and ('bert' not in n and 'decoderEmbedding' not in n)], 'weight_decay_rate': 0.0, 'lr': args.decoder_learning_rate},
        ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.encoder_learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    if args.do_train:

        eval_loss, eval_ppl = eval_the_model(args, model, eval_data)
        best_ppl = eval_ppl
        log_path = os.path.join(args.output_dir, 'log.txt.'+args.postfix)
        with open(log_path, 'w') as f:
            f.write('Before train, the average loss on val set is: ' + str(eval_loss.float()) + ' and the average ppl on val set is: ' + str(eval_ppl))
            f.write('\n\n')
        
        model.train()

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            train_loss = 0
            train_batch = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                # input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                input_ids, input_mask, segment_ids, answer_ids, answer_mask = batch
                loss, _ = model(input_ids, segment_ids, input_mask, answer_ids=answer_ids, answer_mask=answer_mask)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

                train_loss += loss.detach().cpu()
                train_batch += 1

            train_loss /= train_batch
            train_ppl = math.pow(math.e, train_loss)
            eval_loss, eval_ppl = eval_the_model(args, model, eval_data)
            with open(log_path, 'a') as f:
                f.write('In epoch-' + str(epoch) + ' the average loss on train set is: ' + str(train_loss.float()) + ' the average ppl on train set is: ' + str(train_ppl))
                f.write('\n')
                f.write('In epoch-' + str(epoch) + ' the average loss on eval set is: ' + str(eval_loss.float()) + ' the average ppl on eval set is: ' + str(eval_ppl))
                f.write('\n\n')
            if eval_ppl < best_ppl:
                model_save_path = os.path.join(args.output_dir, 'models', 'best_params.pt.'+args.postfix)
                if os.path.exists(model_save_path):
                    os.remove(model_save_path) 
                torch.save(model.state_dict(), model_save_path)
            
 
    if args.do_predict:

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_answers = collections.OrderedDict()
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, answer_ids, answer_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            answer_ids = answer_ids.to(device)

            with torch.no_grad():

                def decode_to_vocab(batch, isContext=False):
                    with torch.cuda.device_of(batch):
                        batch = batch.tolist()
                    batch = [[idx_to_vocab[ind] for ind in ex] for ex in batch]
                    
                    def trim(s, t):
                        sentence = []
                        for w in s:
                            if w == t:
                                break
                            sentence.append(w)
                        return sentence

                    batch = [trim(ex, '[SEP]') for ex in batch]

                    def filter_special(tok):
                        return tok not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EOS]']

                    batch = [filter(filter_special, ex) for ex in batch]

                    return [' '.join(ex) if ex != None else '' for ex in batch]

                loss, outs = model(input_ids, segment_ids, input_mask, answer_ids=answer_ids)

                ground_trurhs = decode_to_vocab(answer_ids)
                decode_answers = decode_to_vocab(outs)
                decode_contexts = decode_to_vocab(input_ids, isContext=True)

                # qas_ids = qas_ids.tolist()
                # for answer_idx, answer_text in zip(qas_ids, decode_answers):
                #     # print(answer_idx)
                #     answer_text = answer_text.replace(" ##", "")
                #     answer_text = answer_text.replace("##", "")
                #     all_answers[id_dict[answer_idx]] = answer_text
        # output_answer_file = os.path.join(args.output_dir, 'predictions', 'predictions.txt.'+args.postfix)
        # with open(output_answer_file, "w") as f:
        #     f.write(json.dumps(all_answers, indent=4) + "\n")

                for i, (context, answer, ground_trurh) in enumerate(zip(decode_contexts, decode_answers, ground_trurhs)):
                    print('context is:\n', context.replace(" ##", ""))
                    print('ground truth is:\n', ground_trurh.replace(" ##", ""))
                    print('answer is:\n', answer.replace(" ##", ""))
                    if i == 1:
                        break


if __name__ == "__main__":
    main()
