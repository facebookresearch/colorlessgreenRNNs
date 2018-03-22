# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import pandas as pd

from collections import defaultdict
import string


def read_paradigms(path):
    """ reads morphological paradigms from a file with token, lemma, tag, morph, freq fields
        returns a simple dict: token -> list of all its analyses and their frequencies """
    d = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            token, lemma, tag, morph, freq = line.split("\t")
            s_m = morph.split("|")
            s_m.sort()
            morph = "|".join(s_m)
            d[token].append((lemma, tag, morph, int(freq)))
    return d


def load_vocab(vocab_file):
    f_vocab = open(vocab_file, "r")
    vocab = {w: i for i, w in enumerate(f_vocab.read().split())}
    f_vocab.close()
    return vocab



def ltm_to_word(paradigms):
    """ converts standard paradigms dict (token -> list of analyses) to a dict (l_emma, t_ag, m_orph -> word)
        (where word in the most frequent form, e.g. between capitalized and non-capitalized Fanno and fanno) """
    #paradigms = read_paradigms("/private/home/gulordava/edouard_data/itwiki//paradigms_UD.txt")

    paradigms_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for w in paradigms:
        for lemma, tag, morph, freq in paradigms[w]:
            paradigms_lemmas[(lemma, tag)][morph][w] = int(freq)

    best_paradigms_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for l, t in paradigms_lemmas:
        for m in paradigms_lemmas[(l, t)]:
            word = sorted(paradigms_lemmas[(l, t)][m].items(), key=lambda x: -x[1])[0][0]
            best_paradigms_lemmas[l][t][m] = word
    return best_paradigms_lemmas


def vocab_freqs(train_data_file, vocab):
    train_data = open(train_data_file).readlines()
    freq_dict = {}
    for w in vocab:
        freq_dict[w] = 0
    for line in train_data:
        for w in line.split():
            if w in vocab:
                freq_dict[w] += 1
    return freq_dict


"""
def number_agreement_data(sents, gold, ltm_paradigms, vocab):
    data = []
    sentence_parts = []

    for sent, g in zip(sents, gold):
        pattern_id, constr_id, sent_id, idx, gold_pos, gold_morph, _, _, _ = g.split()

        if "Number=Plur" in gold_morph:
            correct_number = "plur"
        elif "Number=Sing" in gold_morph:
            correct_number = "sing"
        else:
            continue

        sent_part = sent.split()[:int(idx)]
        #        print(sent_part, gold_pos, gold_morph)
        for lemma, form, form_alt in choose_random_forms(ltm_paradigms, vocab, gold_pos, gold_morph):
            sentence_parts.append(" ".join(sent_part) + " " + form + "\n")
            sentence_parts.append(" ".join(sent_part) + " " + form_alt + "\n")
            data.append((pattern_id, int(constr_id), int(sent_id),
                         lemma, correct_number, form, "correct"))
            data.append((pattern_id, int(constr_id), int(sent_id),
                         lemma, correct_number, form_alt, "wrong"))

    return data, sentence_parts
"""

def plurality(morph):
    if "Number=Plur" in morph:
        return "plur"
    elif "Number=Sing" in morph:
        return "sing"
    else:
        return "none"


def transform_gold(gold):
    data = []
    for g in gold:
        pattern_id, constr_id, sent_id, r_idx, r_pos, r_morph, form, form_alt, lemma, l_idx, l_pos, prefix = g.split(
            "\t")

        correct_number = plurality(r_morph)

        data.append((pattern_id, int(constr_id), int(sent_id), correct_number, form, "correct"))
        data.append((pattern_id, int(constr_id), int(sent_id), correct_number, form_alt, "wrong"))

    return data


def is_attr(word, pos, number, paradigms):
    """ verify whether a word is attractor, that is of tag *pos* and of the number opposite of *number* """
    if not paradigms[word]:
        return False
    max_freq = max([p[3] for p in paradigms[word]])
    for lemma, tag, morph, freq in paradigms[word]:
        # a word can have different tags (be ambiguous)
        # we filter out tags which are very infrequent (including wrong tags for functional words)
        if freq < max_freq / 10:
            continue
        if tag == pos and plurality(morph) != "none" and plurality(morph) != number:
            return True
    return False


def extract_sent_features(sents, gold, vocab, paradigms):
    """ Extracting some features of the construction and the sentence for data analysis """
    paradigms_word_tag = defaultdict(list)
    for w in paradigms:
        for lemma, tag, morph, freq in paradigms[w]:
            paradigms_word_tag[w].append(tag)

    df_sents = []
    constr_id_unk = []
    n_attractors = []
    punct = []
    for s, g in zip(sents, gold):
        pattern_id, constr_id, sent_id, r_idx, r_pos, r_morph, form, form_alt, lemma, l_idx, l_pos, prefix = g.split("\t")
        sent_id = int(sent_id)
        r_idx = int(r_idx)
        l_idx = int(l_idx)
        s_lm = " ".join([w if w in vocab else "<unk>" for w in s.split()[:r_idx]])
        n_unk = len([w for w in s.split()[:r_idx] if w not in vocab ])

        if sent_id == 0:
            constr_id_unk.append((pattern_id, int(constr_id), n_unk))
            number = plurality(r_morph)
            #print(r_morph, number)
            attrs = [w for w in s.split()[l_idx + 1:r_idx] if is_attr(w, l_pos, number, paradigms)]
            n_attractors.append((pattern_id, int(constr_id), len(attrs)))
            #punct.append((pattern_id, int(constr_id), "PUNCT" in pos_seq))
            punct.append((pattern_id, int(constr_id), any(p in prefix.split() for p in string.punctuation)))
            #print(s_lm)
            #print(attrs)

        n_unk = s_lm.count("<unk>")
        len_prefix = len(s_lm.split())
        len_context = r_idx - l_idx
        df_sents.append((pattern_id, int(constr_id), int(sent_id), s.strip(), s_lm, n_unk, len_context, len_prefix))

    df_sents = pd.DataFrame(df_sents, columns = ["pattern_id","constr_id", "sent_id", "sent", "prefix", "n_unk","len_context","len_prefix"])
    #print(constr_id_unk)
    unk = pd.DataFrame(constr_id_unk, columns=["pattern_id", "constr_id", "n_unk_original"])
    attr = pd.DataFrame(n_attractors, columns=["pattern_id","constr_id","n_attr"])
    punct = pd.DataFrame(punct, columns=["pattern_id","constr_id","punct"])
    df_sents = df_sents.merge(unk, on=["pattern_id", "constr_id"])
    df_sents = df_sents.merge(attr, on=["pattern_id", "constr_id"])
    df_sents = df_sents.merge(punct, on=["pattern_id", "constr_id"])
    return df_sents