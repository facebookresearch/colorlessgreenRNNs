#!/usr/bin/env bash
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash

# path to a treebank
# should be preprocessed for English and Hebrew
# you can download original UD treebanks from http://universaldependencies.org/, we used 2.0 version
treebank="../data/treebanks/en-ud-train.conllu_processed"
lang="English"

# path to the training LM data, including vocab.txt, e.g.
lm_data="../data/lm/$lang/"

mkdir -p tmp/$lang

# extracting word forms, lemmas and morphological features
echo "== collect paradigms =="
#python data/collect_paradigms.py --input $treebank --output tmp/$lang/paradigms.txt --min_freq 0

echo "== extract agreement constructions =="
# extracting patterns which correspond to agreement relation
#python syntactic_testsets/extract_dependency_patterns.py --treebank $treebank --output tmp/$lang/ --features Number --vocab $lm_data/vocab.txt --paradigms tmp/$lang/paradigms.txt

echo "== generate test set =="
# given a list of patterns, grep corresponding instances which satisfy generation conditions 
python syntactic_testsets/generate_nonsense.py --treebank $treebank --paradigms tmp/$lang/paradigms.txt \
                                               --vocab $lm_data/vocab.txt --patterns tmp/$lang/patterns.txt \
                                               --output ../data/agreement/$lang/generated \
                                               --lm_data $lm_data   # for estimation of token probabilities from LM training data


