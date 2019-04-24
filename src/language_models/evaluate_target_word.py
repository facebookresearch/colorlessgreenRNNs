# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser(description='Mask-based evaluation: extracts softmax vectors for specified words')

parser.add_argument('--data', type=str,
                    help='location of the data corpus for LM training')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--path', type=str, help='path to test file (text) gold file (indices of words to evaluate)')
parser.add_argument('--suffix', type=str, help='suffix for generated output files which will be saved as path.output_suffix')
args = parser.parse_args()


def evaluate(data_source, mask):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            _, targets_mask = get_batch(mask, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)

            output_candidates_probs(output_flat, targets, targets_mask)

            hidden = repackage_hidden(hidden)

    return total_loss.item() / (len(data_source) - 1)


def output_candidates_probs(output_flat, targets, mask):
    log_probs = F.log_softmax(output_flat, dim=1)

    log_probs_np = log_probs.cpu().numpy()
    subset = mask.cpu().numpy().astype(bool)

    for scores, correct_label in zip(log_probs_np[subset], targets.cpu().numpy()[subset]):
        #print(idx2word[correct_label], scores[correct_label])
        f_output.write("\t".join(str(s) for s in scores) + "\n")


def create_target_mask(test_file, gold_file, index_col):
    sents = open(test_file, "r").readlines()
    golds = open(gold_file, "r").readlines()
    #TODO optimize by initializaing np.array of needed size and doing indexing
    targets = []
    for sent, gold in zip(sents, golds):
        # constr_id, sent_id, word_id, pos, morph
        target_idx = int(gold.split()[index_col])
        len_s = len(sent.split(" "))
        t_s = [0] * len_s
        t_s[target_idx] = 1
        #print(sent.split(" ")[target_idx])
        targets.extend(t_s)
    return np.array(targets)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    print("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = 1
seq_len = 20

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)
#print("Vocab size", vocab_size)
print("Computing probabilities for target words")

# assuming the mask file contains one number per line indicating the index of the target word
index_col = 0

mask = create_target_mask(args.path + ".text", args.path + ".eval", index_col)
mask_data = batchify(torch.LongTensor(mask), eval_batch_size, args.cuda)
test_data = batchify(dictionary_corpus.tokenize(dictionary, args.path + ".text"), eval_batch_size, args.cuda)

f_output = open(args.path + ".output_" + args.suffix, 'w')
evaluate(test_data, mask_data)
print("Probabilities saved to", args.path + ".output_" + args.suffix)
f_output.close()


