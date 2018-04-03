# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd

import os, sys

from syntactic_testsets.utils import load_vocab

def lstm_probs(output, gold, w2idx):
    data = []
    for scores, g in zip(output, gold):
        scores = scores.split()
        form, form_alt = g.split("\t")[6:8]

        prob_correct = float(scores[w2idx[form]])
        prob_wrong = float(scores[w2idx[form_alt]])

        data.append(prob_correct)
        data.append(prob_wrong)

    return data

lang = sys.argv[1]
model = sys.argv[2]

path_repo = "../data"
path_test_data = path_repo + "/agreement/" + lang + "/generated"
path_output = path_repo + "/agreement/" + lang + "/generated.output_"
path_lm_data = path_repo + "/lm/" + lang

gold = open(path_test_data + ".gold").readlines()
sents = open(path_test_data + ".text").readlines()
data = pd.read_csv(path_test_data + ".tab",sep="\t")

vocab = load_vocab(path_lm_data + "/vocab.txt")

# getting softmax outputs and the probabilities for pairs of test forms
#print("Assembling probabilities for the choice forms")
outputs = {}
probs = pd.DataFrame([])

if os.path.isfile(path_output + model):
    #print(model)
    output = open(path_output  + model).readlines()
    #print(len(output))
    data[model] = lstm_probs(output, gold, vocab)


### If you want to save table with target singular and plural form probabilities uncomment these lines and change the path ###
#path_result = path_repo + "/results/" + lang + "/" + model + ".tab"
#print("The target singular and plural form probabilities are saved in", path_result)
#data.to_csv(path_result, sep="\t", index=False)

#### Computing accuracy for the model (and frequency baseline) ####
if "freq" in data:
    models = [model, "freq"]
else:
    models = [model]

fields = ["pattern","constr_id","sent_id","n_attr","punct","len_prefix","len_context","sent","correct_number","type"]
wide_data = data[fields + ["class"] + models].pivot_table(columns=("class"), values=models, index=fields)

for model in models:
    correct = wide_data.loc[:, (model, "correct")]
    wrong = wide_data.loc[:, (model, "wrong")]
    wide_data[(model, "acc")] = (correct > wrong)*100

t = wide_data.reset_index()


a = t.groupby("type").agg({(m,"acc"):"mean" for m in models})

print("Accuracy overall\n", a)

a = pd.concat([t[t.type=="original"].groupby("pattern").agg({(m, "acc"): "mean" for m in models}).rename(columns={'acc': 'orig'}),
               t[t.type=="generated"].groupby("pattern").agg({(m, "acc"): "mean" for m in models}).rename(columns={'acc': 'gen'})], axis=1)

print()
print("Accuracy by pattern\n", a)
