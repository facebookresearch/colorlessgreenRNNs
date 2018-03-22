# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import sys
from utils import read_paradigms, load_vocab, extract_sent_features, transform_gold, vocab_freqs

import pandas as pd


lang = sys.argv[1]

path_test_data = "/private/home/gulordava/colorlessgreen/data/agreement/" + lang + "/generated"
path_lm_data = "/private/home/gulordava/colorlessgreen/data/lm/" + lang

if lang == "English":
    path_paradigms = "/private/home/gulordava/edouard_data/enwiki/paradigms_UD.txt"
if lang == "Italian":
    path_paradigms = "/private/home/gulordava/edouard_data/itwiki/paradigms_UD.txt"
if lang == "Italian_srnn":
    path_paradigms = "/private/home/gulordava/edouard_data/itwiki/paradigms_UD.txt"
if lang == "Russian":
    path_paradigms = "/private/home/gulordava/edouard_data/ruwiki/paradigms_UD.txt"
if lang == "Hebrew":
    path_paradigms = "/private/home/gulordava/edouard_data/hewiki/p2"

gold = open(path_test_data + ".gold").readlines()
sents = open(path_test_data + ".text").readlines()

paradigms = read_paradigms(path_paradigms)

output = []
vocab = load_vocab(path_lm_data + "/vocab.txt")

data = transform_gold(gold)
data = pd.DataFrame(data, columns=["pattern_id", "constr_id", "sent_id", "correct_number", "form", "class"])
data.loc[data.sent_id == 0, "type"] = "original"
data.loc[data.sent_id > 0, "type"] = "generated"

# getting simpler pattern labels
patterns = {p: "__".join(p.split("!")[:2]) for p in set(data.pattern_id)}
data["pattern"] = data["pattern_id"].map(patterns)

df_sents = extract_sent_features(sents, gold, vocab, paradigms)
full_df = data.merge(df_sents, on=["pattern_id", "constr_id", "sent_id"])

freq_dict = vocab_freqs(path_lm_data + "/train.txt", vocab)
full_df["freq"] = full_df["form"].map(freq_dict)
fields = ["pattern", "constr_id", "sent_id", "correct_number", "form", "class", "type", "prefix", "n_attr",
              "punct", "freq", "len_context", "len_prefix", "sent"]

full_df[fields].to_csv(path_test_data + ".tab", sep="\t", index=False)