import argparse
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from collections import OrderedDict
from analysis import rocstories as rocstories_analysis
from datasets import msmarco
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, encode_dataset_msmarco, iter_data, iter_data_msmarco,
                   ResultLogger, make_path)
from loss import MultipleChoiceLossCompute

from parallel import DataParallelModel, DataParallelCriterion

# def transform_msmarco(queries, passages, answers=None):
#     n_batch = len(queries)
#     xmb = np.zeros((n_batch, 10, n_ctx, 2), dtype=np.int32)
#     mmb = np.zeros((n_batch, 10, n_ctx), dtype=np.float32)
#     start = encoder['_start_']
#     delimiter = encoder['_delimiter_']
#     for i in range(n_batch):
#         for j in range(10):
#             x = [start]+passages[j][i][:p_max_len]+[delimiter]+queries[i][:q_max_len]+[clf_token]
#             length = len(x)
#             xmb[i, j, :length, 0] = x
#             mmb[i, j, :length] = 1
#     xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
#     return xmb, mmb

def transform_msmarco_batch(queries, passages, targets, isValid=False, isInference=False):
    n_batch = len(queries)
    max_length = get_max_length(queries, passages, targets, isInference)
    max_length = max_length+3 if isInference else max_length+4
    xmb = np.zeros((n_batch, max_length, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, max_length), dtype=np.float32)
    # lengths = []
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    sos = encoder['_sos_']
    eos = encoder['_eos_']
    for i in range(n_batch):
        x = [start]+passages[i][:p_max_len]+[delimiter]+queries[i][:q_max_len]+[sos]+targets[i]+[eos]
        if isInference:
            x = [start]+passages[i][:p_max_len]+[delimiter]+queries[i][:q_max_len]+[sos]
        # if question_first:
        #     x = [start]+queries[i][:q_max_len]+[delimiter]+passages[i][:p_max_len]+[sos]+targets[i]+[eos]
        length_all = len(x)
        length_inp = len([start]+queries[i][:q_max_len]+[delimiter]+passages[i][:p_max_len])
        xmb[i, :length_all, 0] = x
        mmb[i, :length_all] = 1
        mmb[i, :length_inp] = 0 if isValid else lm_coef
        # lengths.append(length)
    xmb[:, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+max_length)
    return xmb, mmb

def get_max_length(queries, passages, targets, isInference=False):
    n_batch = len(queries)
    lengths = []
    for i in range(n_batch):
        if isInference:
            lengths.append(len(queries[i][:40]) + len(passages[i][:200])) #+ len(targets[i]))
        else:
            lengths.append(len(queries[i][:40]) + len(passages[i][:200]) + len(targets[i]))
    return max(lengths)

def iter_apply_msmarco(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=True, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.float).to(device)
            MMB = torch.tensor(mmb).to(device)
            ret = dh_model(XMB)
            clf_logits = []
            for item in ret:
                clf_logits.append(item[1])
            clf_logits = torch.cat(clf_logits, dim = 1)
            clf_logits = clf_logits.view(n, -1)
            logits.append(clf_logits.to("cpu").numpy())
        logits = np.concatenate(logits, 0)
    return logits #, cost

def iter_apply_msmarco_batch(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    total_n = 0
    with torch.no_grad():
        dh_model.eval()
        for queries, passages, targets in iter_data_msmarco(Xs, Ms, Ys, n_batch=n_batch_train, truncate=True, verbose=True):
            total_n += len(queries)
            xmb, mmb = transform_msmarco_batch(queries, passages, targets, isValid = True)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            # YMB = torch.tensor(ymb, dtype=torch.float).to(device)
            MMB = torch.tensor(mmb).to(device)
            lm_logits = dh_model(XMB)
            # print(type(lm_logits))
            # print(lm_logits.size())
            xmb_shifted = XMB[:, 1:, 0].contiguous().view(-1)
            mmb = MMB.view(-1, MMB.size(1))
            lm_loss = criterion_lm(lm_logits, xmb_shifted)
            lm_loss = lm_loss.view(XMB.size(0), XMB.size(1) - 1)
            lm_loss = lm_loss * mmb[:, 1:]
            # lm_loss = lm_loss.sum(1) / torch.sum(mmb > 0, 1)
            lm_loss = lm_loss.sum(1) # / (mmb > 0).sum(1).float()
            dev_loss = lm_loss.sum()
            # clf_logits_softmax = []
            # clf_logits_log_softmax = []
            # for item in ret:
            #     clf_logits_softmax.append(F.softmax(item[1], dim=1))
            #     clf_logits_log_softmax.append(F.log_softmax(item[1], dim=1))
            # try:
            #     clf_losses = criterion_clf(clf_logits_log_softmax, YMB)
            # except TypeError:
            #     print('\n Error YMB is', YMB)
            #     print('Error YMB type is', type(YMB))
            #     clf_losses = np.array([0])
            # clf_logits_softmax = torch.cat(clf_logits_softmax, dim = 1)
            # clf_logits_softmax = clf_logits_softmax.view(n, -1)
            # logits.append(clf_logits_softmax.to("cpu").numpy())
            # cost += clf_losses.sum().item()
            cost += dev_loss.item()
        # logits = np.concatenate(logits, 0)
    return cost/total_n

def log_msmarco():
    global best_score
    print("Logging")
    # dev_logits, cost = iter_apply_msmarco_batch(dev_queries, dev_passages, dev_targets)
    cost = iter_apply_msmarco_batch(dev_queries, dev_passages, dev_targets)
    # print('dev_targets is:', dev_targets[:10])
    # print('dev_logits is: ', dev_logits[:10])
    # print('dev_logits is: ', dev_logits[:10]>threshold)
    # dev_acc = accuracy_score(np.array(dev_targets), dev_logits>threshold)*100.
    # hit_acc_1, hit_acc_2, hit_acc_3 = log_highest_score(np.array(dev_targets), dev_logits)
    # hit_acc_1, hit_acc_2, hit_acc_3 = hit_acc_1*100., hit_acc_2*100., hit_acc_3*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, cost=cost)
    print('%d %d %.2f'%(n_epochs, n_updates, cost))
    if submit:
        score = cost
        if score < best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params_gen_model')
            torch.save(dh_model.state_dict(), make_path(path))

def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)

def print_inp(inp):
    inp = inp[0]
    pre_token  = ''
    for item in inp:
        token = text_encoder.decoder[item] #.replace('</w>', '')
        if token[-4:] == '</w>':
            pre_token += token.replace('</w>', '')
            print(pre_token, end=' ')
            pre_token = ''
        else:
            pre_token += token
    print()

def inference_msmarco(test_passages, test_queries, test_targets):
    with torch.no_grad():
        dh_model.eval()
        for queries, passages, targets in iter_data_msmarco(test_queries, test_passages, test_targets, n_batch=n_gpu, truncate=True, verbose=True):
            n = len(queries)
            print('\n\npassage is:')
            print_inp(passages)
            print('\nquery is:')
            print_inp(queries)
            print('\nground truth is:')
            print_inp(targets)
            xmb, _ = transform_msmarco_batch(queries, passages, targets, isInference = True)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            print('\nanswer is:')
            pre_token = ''
            for _ in range(4):
                # print('XMB size is', XMB.size())                
                lm_logits = dh_model(XMB)
                # print('lm_logits size is', lm_logits.size())                
                lm_logits = lm_logits.view(-1, XMB.size(1)-1, lm_logits.size(-1))

                # print('lm_logits size is', lm_logits.size())
                # # exit(0)
                # print('before add the mask:\n', lm_logits[:, -1, :])
                pos_emb_mask = torch.zeros(1, 1, vocab).to(device)
                pos_emb_mask[:, :, -n_ctx-4:] = -1e12
                # print('mask is :', pos_emb_mask)
                # print('mask size is :', pos_emb_mask.size())
                lm_logits = lm_logits + pos_emb_mask
                # print('after add the mask:\n', lm_logits[:, -1, :])
                
                # print('lm_logits size is', lm_logits.size())
                lm_logits = F.softmax(lm_logits, dim=-1)
                # print('after the softmax:\n', lm_logits[:, -1, :], lm_logits[:, -1, :][0][0], lm_logits[:, -1, :][0][-1])
                # print('lm_logits size is :', lm_logits.size())
                # print('lm_logits[:, -1, :] is:', lm_logits[:, -1, :].size())
                # exit(0)
                next_idx = torch.multinomial(lm_logits[:, -1, :], 1)
                next_token = text_encoder.decoder[next_idx.item()] #.replace('</w>', '')
                
                if next_token[-4:] == '</w>':
                    pre_token += next_token.replace('</w>', '')
                    print(pre_token, end=' ')
                    pre_token = ''
                else:
                    pre_token += next_token
                
                if next_token == encoder['_eos_']:
                    break
                # print(next_token, end=' ')
                XMB = append_batch(XMB, next_idx)
            print('\n')

def log_highest_score(dev_targets, dev_logits):
    # test whether the highest score hits the target passage.
    n_answerable = 0
    n_hits_1 = 0
    n_hits_2 = 0
    n_hits_3 = 0
    for t, l in zip(dev_targets, dev_logits):
        if sum(t) == 0:
            continue
        n_answerable += 1
        ids = np.argsort(-l)
        
        if t[ids[0]] == 1:
            n_hits_1 += 1
        if t[ids[0]] == 1 or t[ids[1]] == 1:
            n_hits_2 += 1
        if t[ids[0]] == 1 or t[ids[1]] == 1 or t[ids[2]] == 1:
            n_hits_3 += 1

    return n_hits_1/n_answerable, n_hits_2/n_answerable, n_hits_3/n_answerable 

def run_epoch():
    i = 0
    for queries, passages, targets in iter_data_msmarco(train_queries, train_passages, train_targets, n_batch=n_batch_train, truncate=True, verbose=True, training=True):
        global n_updates
        dh_model.train()
        xmb, mmb = transform_msmarco_batch(queries, passages, targets)
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        # YMB = torch.tensor(targets, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits = dh_model(XMB)

        xmb_shifted = XMB[:, 1:, 0].contiguous().view(-1)
        mmb = MMB.view(-1, MMB.size(1))
        lm_loss = criterion_lm(lm_logits, xmb_shifted)
        lm_loss = lm_loss.view(XMB.size(0), XMB.size(1) - 1)
        lm_loss = lm_loss * mmb[:, 1:]
        lm_loss = lm_loss.sum(1) # / (mmb > 0).sum(1).float()
        # clf_loss = criterion_clf(clf_logits, YMB)

        # if lm_coef > 0:
        train_loss = lm_loss.sum() #* lm_coef

        # train_loss = train_loss   ##TODO accumulating gradients
        train_loss.backward()
        model_opt.step()
        model_opt.zero_grad()
        n_updates += 1
        i += 1

def sort_by_length(queries, passages, target):
    num_exs = len(queries)
    lengths = []
    for i in range(num_exs):
        # lengths.append(len(queries[i][:40]) + max([len(passages[j][i][:200]) for j in range(10)]))
        lengths.append(len(queries[i][:40]) + len(passages[i][:200]) + len(target[i]))
    lengths = np.array(lengths)
    indices = np.argsort(lengths)
    passages = list(np.array(passages)[indices])
    queries = list(np.array(queries)[indices])
    # for i in range(10):
    #     passages[i] = list(np.array(passages[i])[indices])
    target = list(np.array(target)[indices])
    return queries, passages, target

def save_preprocessed_file(train_queries, train_passages, train_targets, dev_queries, dev_passages, dev_targets):
    with open(args.save_dir + '/train_queries_gen_model.json', 'w') as fp:
        json.dump(train_queries, fp)
    with open(args.save_dir + '/train_passages_gen_model.json', 'w') as fp:
        json.dump(train_passages, fp)
    with open(args.save_dir + '/train_targets_gen_model.json', 'w') as fp:
        json.dump(train_targets, fp)
    with open(args.save_dir + '/dev_queries_gen_model.json', 'w') as fp:
        json.dump(dev_queries, fp)
    with open(args.save_dir + '/dev_passages_gen_model.json', 'w') as fp:
        json.dump(dev_passages, fp)
    with open(args.save_dir + '/dev_targets_gen_model.json', 'w') as fp:
        json.dump(dev_targets, fp)

def load_preprocessed_file():
    with open(args.save_dir + '/train_queries_gen_model.json', 'r') as fp:
        train_queries = json.load(fp)
    with open(args.save_dir + '/train_passages_gen_model.json', 'r') as fp:
        train_passages = json.load(fp)
    with open(args.save_dir + '/train_targets_gen_model.json', 'r') as fp:
        train_targets = json.load(fp)
    with open(args.save_dir + '/dev_queries_gen_model.json', 'r') as fp:
        dev_queries = json.load(fp)
    with open(args.save_dir + '/dev_passages_gen_model.json', 'r') as fp:
        dev_passages = json.load(fp)
    with open(args.save_dir + '/dev_targets_gen_model.json', 'r') as fp:
        dev_targets = json.load(fp)
    return (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets)

def filter_dataset(queries, passages, targets):
    # Keep data with answers less than 4 words
    print('filter the dataset ...')
    n = len(queries)
    new_queries = []
    new_passages = []
    new_targets = []
    for i in range(n):
        if len(targets[i]) <= 4:
            new_queries.append(queries[i])
            new_passages.append(passages[i])
            new_targets.append(targets[i])
    return new_queries, new_passages, new_targets

argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--threshold', type=int, default=0.36)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--training_portion', type=float, default=0)
    parser.add_argument('--question_first', action='store_true')
    parser.add_argument('--generative', action='store_true')
    parser.add_argument('--load_pretrained_model', action='store_true')
    parser.add_argument('--inference', action='store_true')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir
    lm_coef = args.lm_coef
    threshold = args.threshold
    accumulation_steps = args.accumulation_steps
    question_first = args.question_first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print('n_vocab is:', n_vocab)
    print("Encoding dataset...")
  
    # (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets) = encode_dataset_msmarco(*msmarco(data_dir), encoder=text_encoder)
    # save_preprocessed_file(train_queries, train_passages, train_targets, dev_queries, dev_passages, dev_targets)
    (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets) = load_preprocessed_file()
    train_queries, train_passages, train_targets = filter_dataset(train_queries, train_passages, train_targets)
    dev_queries, dev_passages, dev_targets = filter_dataset(dev_queries, dev_passages, dev_targets)
    # print_inp(train_queries[1:])
    # print_inp(train_passages[1:])
    # print_inp(train_targets[1:])
    # exit(0)
    
    if args.training_portion > 0:
        train_length = len(train_queries)
        train_portion = int(train_length * args.training_portion)
        random_idx = []
        for i in range(train_portion):
            random_idx.append(random.randint(0, train_length))
        random_idx = np.array(random_idx)
        train_passages = list(np.array(train_passages)[random_idx])
        train_queries = list(np.array(train_queries)[random_idx])
        # train_passages = [list(np.array(train_passage)[random_idx]) for train_passage in train_passages]
        train_targets = list(np.array(train_targets)[random_idx])
        
        dev_length = len(dev_queries)
        dev_portion = int(dev_length * args.training_portion)
        random_idx = []
        for i in range(dev_portion):
            random_idx.append(random.randint(0, dev_length))
        random_idx = np.array(random_idx)
        dev_passages = list(np.array(dev_passages)[random_idx])        
        dev_queries = list(np.array(dev_queries)[random_idx])
        # dev_passages = [list(np.array(dev_passage)[random_idx]) for dev_passage in dev_passages]
        dev_targets = list(np.array(dev_targets)[random_idx])

    test_length = len(dev_queries)
    random_idx = []
    for i in range(16):
        random_idx.append(random.randint(0, test_length))
    random_idx = np.array(random_idx)
    test_passages = list(np.array(dev_passages)[random_idx])        
    test_queries = list(np.array(dev_queries)[random_idx])
    test_targets = list(np.array(dev_targets)[random_idx])
    (train_queries, train_passages, train_targets) = sort_by_length(train_queries, train_passages, train_targets)
    (test_queries, test_passages, test_targets) = sort_by_length(test_queries, test_passages, test_targets)

    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_sos_'] = len(encoder)
    encoder['_eos_'] = len(encoder)
    # clf_token = encoder['_classify_']
    n_special = 4
    # max_len = n_ctx // 2 - 2
    q_max_len = 40
    p_max_len = 200

    t = []
    train_exs = len(train_queries)
    # print('train_exs is', train_exs)
    for i in range(train_exs):
        # tmp_length = []
        x1 = train_queries[i]
        x2 = train_passages[i]
        x3 = train_targets[i]
        tmp_length = len(x1[:q_max_len]) + len(x2[:p_max_len]) + len(x3)
        t.append(tmp_length)

    dev_exs = len(dev_queries)
    # print('dev_exs is', dev_exs)
    for i in range(dev_exs):
        tmp_length = []
        x1 = dev_queries[i]
        x2 = dev_passages[i]
        x3 = dev_targets[i]
        tmp_length = len(x1[:q_max_len]) +len(x2[:p_max_len]) + len(x3)
        t.append(tmp_length)

    # _t = []
    # test_exs = len(test_queries)
    # # print('dev_exs is', dev_exs)
    # for i in range(test_exs):
    #     tmp_length = []
    #     x1 = test_queries[i]
    #     x2 = test_passages[i]
    #     x3 = test_targets[i]
    #     tmp_length = len(x1[:q_max_len]) +len(x2[:p_max_len]) + len(x3)
    #     _t.append(tmp_length)
    # print(_t)
    # exit(0)
    # print(len(t))
    # print(max(t))
    # exit(0)
    n_ctx = min(max(t) + 4, n_ctx)
    print('n_ctx is: ', n_ctx)
    vocab = n_vocab + n_special + n_ctx

    n_train = len(train_queries)
    n_valid = len(dev_queries)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = DoubleHeadModel(args, None, 'msmarco_gen', vocab, n_ctx)
    
    criterion_lm = nn.CrossEntropyLoss(reduce=False)
    # criterion_clf = nn.KLDivLoss()
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    
    if args.load_pretrained_model:
        path = os.path.join(save_dir, desc, 'best_params_gen_model')
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        dh_model.load_state_dict(new_state_dict)
    else:
        load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

    dh_model.to(device)
    
    dh_model = DataParallelModel(dh_model)
    criterion_lm = DataParallelCriterion(criterion_lm)
    # criterion_clf = DataParallelCriterion(criterion_clf)

    n_updates = 0
    n_epochs = 0
    best_score = 1e12

    if args.inference:
        inference_msmarco(test_passages, test_queries, test_targets)
    else:
        for i in range(args.n_iter):
            if i == 0:
                log_msmarco()
            print("running epoch", i)
            run_epoch()
            n_epochs += 1
            # log(save_dir, desc)
            log_msmarco()
    
        # inference_msmarco(test_passages, test_queries, test_targets)

    # if submit:
    #     path = os.path.join(save_dir, desc, 'best_params')
    #     dh_model.load_state_dict(torch.load(path))
    #     predict(dataset, args.submission_dir)
    #     if args.analysis:
    #         rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
    #                             os.path.join(log_dir, 'rocstories.jsonl'))
