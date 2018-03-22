# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from collections import defaultdict

from data import data_utils

parser = argparse.ArgumentParser(description='Reading and processing a large gzip file')

parser.add_argument('--input', type=str, required=True,
                    help='Input path (in a column CONLL UD format)')
parser.add_argument('--output', type=str, required=True, help="Output file name")
parser.add_argument('--nwords', type=int, default='100000000', required=False,
                    help='How many words to process')
parser.add_argument('--min_freq', type=int, default='5', required=False,
                    help='Minimal frequency of paradigm to be included in the dictionary')
args = parser.parse_args()

nwords = 0
paradigms = defaultdict(int)
for line in data_utils.read(args.input):
    if line.strip() == "" or len(line.split("\t")) < 2:
        continue
    else:
        fields = line.split("\t")
        if fields[1].isalpha():
            paradigms[(fields[1], fields[2], fields[3], fields[5])] += 1
        nwords += 1
    if nwords > args.nwords:
        break

with open(args.output, 'w') as f:
    for p in paradigms:
        if paradigms[p] > args.min_freq:
            f.write("\t".join(el for el in p) + "\t" + str(paradigms[p]) + "\n")
    f.close()

