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

from analysis import rocstories as rocstories_analysis
from datasets import msmarco
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, encode_dataset_msmarco, iter_data, iter_data_msmarco,
                   ResultLogger, make_path)
from loss import MultipleChoiceLossCompute

from parallel import DataParallelModel, DataParallelCriterion

def transform_msmarco(queries, passages, answers=None):
    n_batch = len(queries)
    xmb = np.zeros((n_batch, 10, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 10, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i in range(n_batch):
        for j in range(10):
            x = [start]+passages[j][i][:p_max_len]+[delimiter]+queries[i][:q_max_len]+[clf_token]
            length = len(x)
            xmb[i, j, :length, 0] = x
            mmb[i, j, :length] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def transform_msmarco_batch(queries, passages, answers=None):
    n_batch = len(queries)
    max_length = get_max_length(queries, passages) + 3
    xmb = np.zeros((n_batch, 10, max_length, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 10, max_length), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i in range(n_batch):
        for j in range(10):
            x = [start]+passages[j][i][:p_max_len]+[delimiter]+queries[i][:q_max_len]+[clf_token]
            if question_first:
                x = [start]+queries[i][:q_max_len]+[delimiter]+passages[j][i][:p_max_len]+[clf_token]
            length = len(x)
            xmb[i, j, :length, 0] = x
            mmb[i, j, :length] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+max_length)
    return xmb, mmb

def get_max_length(queries, passages, answers=None):
    n_batch = len(queries)
    lengths = []
    for i in range(n_batch):
        lengths.append(len(queries[i][:40]) + max([len(passages[j][i][:200]) for j in range(10)]))
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
    with torch.no_grad():
        dh_model.eval()
        for queries, passages, ymb in iter_data_msmarco(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(queries)
            xmb, mmb = transform_msmarco_batch(queries, passages)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.float).to(device)
            MMB = torch.tensor(mmb).to(device)
            ret = dh_model(XMB)
            clf_logits_softmax = []
            clf_logits_log_softmax = []
            for item in ret:
                clf_logits_softmax.append(F.softmax(item[1], dim=1))
                clf_logits_log_softmax.append(F.log_softmax(item[1], dim=1))
            try:
                clf_losses = criterion_clf(clf_logits_log_softmax, YMB)
            except TypeError:
                print('\n Error YMB is', YMB)
                print('Error YMB type is', type(YMB))
                clf_losses = np.array([0])
            clf_logits_softmax = torch.cat(clf_logits_softmax, dim = 1)
            clf_logits_softmax = clf_logits_softmax.view(n, -1)
            logits.append(clf_logits_softmax.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost

def log_msmarco():
    global best_score
    print("Logging")
    dev_logits, cost = iter_apply_msmarco_batch(dev_queries, dev_passages, dev_targets)
    print('dev_targets is:', dev_targets[:10])
    print('dev_logits is: ', dev_logits[:10])
    print('dev_logits is: ', dev_logits[:10]>threshold)
    dev_acc = accuracy_score(np.array(dev_targets), dev_logits>threshold)*100.
    hit_acc_1, hit_acc_2, hit_acc_3 = log_highest_score(np.array(dev_targets), dev_logits)
    hit_acc_1, hit_acc_2, hit_acc_3 = hit_acc_1*100., hit_acc_2*100., hit_acc_3*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, cost=cost, dev_acc=dev_acc, hit_acc_1=hit_acc_1, hit_acc_2=hit_acc_2, hit_acc_3=hit_acc_3)
    print('%d %d %.2f %.2f %.2f %.2f %.2f'%(n_epochs, n_updates, cost, dev_acc, hit_acc_1, hit_acc_2, hit_acc_3))
    if submit:
        score = dev_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))

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
    for queries, passages, ymb in iter_data_msmarco(train_queries, train_passages, train_targets, n_batch=n_batch_train, truncate=True, verbose=True, training=True):
        global n_updates
        dh_model.train()
        xmb, mmb = transform_msmarco_batch(queries, passages)
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.float).to(device)
        MMB = torch.tensor(mmb).to(device)
        ret = dh_model(XMB)
        lm_logits, clf_logits = [], []
        for item in ret:
            lm_logits.append(item[0])
            clf_logits.append(item[1])
        xmb_shifted = XMB[:, :, 1:, 0].contiguous().view(-1)
        mmb = MMB.view(-1, MMB.size(2))
        lm_loss = criterion_lm(lm_logits, xmb_shifted)
        lm_loss = lm_loss.view(XMB.size(0) * XMB.size(1), XMB.size(2) - 1)
        lm_loss = lm_loss * mmb[:, 1:]
        lm_loss = lm_loss.sum(1) / torch.sum(mmb[:, 1:], 1)
        clf_loss = criterion_clf(clf_logits, YMB)

        if lm_coef > 0:
            train_loss = clf_loss.sum() + lm_coef * lm_loss.sum()

        train_loss = train_loss   ##TODO accumulating gradients
        train_loss.backward()
        model_opt.step()
        model_opt.zero_grad()
        n_updates += 1
        i += 1

def sort_by_length(queries, passages, target):
    num_exs = len(queries)
    lengths = []
    for i in range(num_exs):
        lengths.append(len(queries[i][:40]) + max([len(passages[j][i][:200]) for j in range(10)]))
    lengths = np.array(lengths)
    indices = np.argsort(lengths)
    queries = list(np.array(queries)[indices])
    for i in range(10):
        passages[i] = list(np.array(passages[i])[indices])
    target = list(np.array(target)[indices])
    return queries, passages, target

def save_preprocessed_file(train_queries, train_passages, train_targets, dev_queries, dev_passages, dev_targets):
    with open(args.save_dir + '/train_queries_para_select.json', 'w') as fp:
        json.dump(train_queries, fp)
    with open(args.save_dir + '/train_passages_para_select.json', 'w') as fp:
        json.dump(train_passages, fp)
    with open(args.save_dir + '/train_targets_para_select.json', 'w') as fp:
        json.dump(train_targets, fp)
    with open(args.save_dir + '/dev_queries_para_select.json', 'w') as fp:
        json.dump(dev_queries, fp)
    with open(args.save_dir + '/dev_passages_para_select.json', 'w') as fp:
        json.dump(dev_passages, fp)
    with open(args.save_dir + '/dev_targets_para_select.json', 'w') as fp:
        json.dump(dev_targets, fp)

def load_preprocessed_file():
    with open(args.save_dir + '/train_queries_para_select.json', 'r') as fp:
        train_queries = json.load(fp)
    with open(args.save_dir + '/train_passages_para_select.json', 'r') as fp:
        train_passages = json.load(fp)
    with open(args.save_dir + '/train_targets_para_select.json', 'r') as fp:
        train_targets = json.load(fp)
    with open(args.save_dir + '/dev_queries_para_select.json', 'r') as fp:
        dev_queries = json.load(fp)
    with open(args.save_dir + '/dev_passages_para_select.json', 'r') as fp:
        dev_passages = json.load(fp)
    with open(args.save_dir + '/dev_targets_para_select.json', 'r') as fp:
        dev_targets = json.load(fp)
    return (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets)


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

    print("Encoding dataset...")
  
    # (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets) = encode_dataset_msmarco(*msmarco(data_dir), encoder=text_encoder)
    # save_preprocessed_file(train_queries, train_passages, train_targets, dev_queries, dev_passages, dev_targets)
    (train_queries, train_passages, train_targets), (dev_queries, dev_passages, dev_targets) = load_preprocessed_file()
    
    if args.training_portion > 0:
        train_length = len(train_queries)
        train_portion = int(train_length * args.training_portion)
        random_idx = []
        for i in range(train_portion):
            random_idx.append(random.randint(0, train_length))
        random_idx = np.array(random_idx)
        train_queries = list(np.array(train_queries)[random_idx])
        train_passages = [list(np.array(train_passage)[random_idx]) for train_passage in train_passages]
        train_targets = list(np.array(train_targets)[random_idx])
        
        dev_length = len(dev_queries)
        dev_portion = int(dev_length * args.training_portion)
        random_idx = []
        for i in range(dev_portion):
            random_idx.append(random.randint(0, dev_length))
        random_idx = np.array(random_idx)
        dev_queries = list(np.array(dev_queries)[random_idx])
        dev_passages = [list(np.array(dev_passage)[random_idx]) for dev_passage in dev_passages]
        dev_targets = list(np.array(dev_targets)[random_idx])
    (train_queries, train_passages, train_targets) = sort_by_length(train_queries, train_passages, train_targets)

    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    # max_len = n_ctx // 2 - 2
    q_max_len = 40
    p_max_len = 200

    t = []
    train_exs = len(train_queries)
    for i in range(train_exs):
        tmp_length = []
        x1 = train_queries[i]
        for j in range(10):
            x2 = train_passages[j][i]
            tmp_length.append(len(x1[:q_max_len]) +len(x2[:p_max_len]))
        t.append(max(tmp_length))

    dev_exs = len(dev_queries)
    for i in range(dev_exs):
        tmp_length = []
        x1 = dev_queries[i]
        for j in range(10):
            x2 = dev_passages[j][i]
            tmp_length.append(len(x1[:q_max_len]) +len(x2[:p_max_len]))
        t.append(max(tmp_length))

    n_ctx = min(max(t) + 3, n_ctx)
    print('n_ctx is: ', n_ctx)
    vocab = n_vocab + n_special + n_ctx

    n_train = len(train_queries)
    n_valid = len(dev_queries)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = DoubleHeadModel(args, clf_token, 'msmarco_para_select', vocab, n_ctx)

    criterion_lm = nn.CrossEntropyLoss(reduce=False)
    criterion_clf = nn.KLDivLoss()
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
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

    dh_model.to(device)

    dh_model = DataParallelModel(dh_model)
    criterion_lm = DataParallelCriterion(criterion_lm)
    criterion_clf = DataParallelCriterion(criterion_clf)

    n_updates = 0
    n_epochs = 0
    if submit:
        path = os.path.join(save_dir, desc, 'best_params_para_selector')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    for i in range(args.n_iter):
        if i == 0:
            log_msmarco()
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        # log(save_dir, desc)
        log_msmarco()
    # if submit:
    #     path = os.path.join(save_dir, desc, 'best_params')
    #     dh_model.load_state_dict(torch.load(path))
    #     predict(dataset, args.submission_dir)
    #     if args.analysis:
    #         rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
    #                             os.path.join(log_dir, 'rocstories.jsonl'))
