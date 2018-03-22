# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
from collections import defaultdict
from random import shuffle

from data import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input file path')
parser.add_argument('--output', type=str, help='Output file path')
parser.add_argument('--output_dir', type=str, help='Output path for training/valid/test sets')
parser.add_argument('--vocab', type=int, default=10000, help="The size of vocabulary, default = 10K")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def create_vocab(path, vocab_size):
    counter = defaultdict(int)
    for line in data_utils.read(path):
        for word in line.replace("\n"," <eos>").split():
            counter[word] += 1

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:vocab_size]
    words = [w for (w, v) in count_pairs]
    print(len(counter), count_pairs[vocab_size - 1])
    w2idx = dict(zip(words, range(len(words))))
    idx2w = dict(zip(range(len(words)), words))
    return w2idx, idx2w

def convert_text(input_path, output_path, vocab):
    with open(output_path, 'w') as output:
        for line in data_utils.read(input_path):
            words = [filter_word(word, vocab) for word in line.replace("\n", " <eos>").split()]
            output.write(" ".join(words) + "\n")
        output.close()

def convert_line(line, vocab):
    return [filter_word(word, vocab) for word in line.replace("\n", " <eos>").split()]

def word_to_idx(word, vocab):
    if word in vocab:
        return vocab[word]
    else:
        return vocab["<unk>"]

def filter_word(word, vocab):
    if word in vocab:
        return word
    else:
        return "<unk>"

def create_corpus(input_path, output_path, vocab):
    """ Split data to create training, validation and test corpus """
    nlines = 0
    f_train = open(output_path + "/train.txt", 'w')
    f_valid = open(output_path + "/valid.txt", 'w')
    f_test = open(output_path + "/test.txt", 'w')

    train = []

    for line in data_utils.read(input_path):
        if nlines % 10 == 0:
            f_valid.write(" ".join(convert_line(line, vocab)) + "\n")
        elif nlines % 10 == 1:
            f_test.write(" ".join(convert_line(line, vocab)) + "\n")
        else:
            train.append(" ".join(convert_line(line, vocab)) + "\n")
        nlines += 1

    shuffle(train)
    f_train.writelines(train)

    f_train.close()
    f_valid.close()
    f_test.close()


w2idx, idx2w = create_vocab(args.input, args.vocab)

#convert_text(args.input, args.output, w2idx)
create_corpus(args.input, args.output_dir, w2idx)

