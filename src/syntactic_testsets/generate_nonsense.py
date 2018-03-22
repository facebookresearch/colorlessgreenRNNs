# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import random

import pandas as pd

import tree_module as tm
from extract_dependency_patterns import grep_morph_pattern
from generate_utils import is_good_form, get_alt_form, match_features, alt_numeral_morph
from utils import read_paradigms, load_vocab, ltm_to_word, extract_sent_features, transform_gold, vocab_freqs


def generate_morph_pattern_test(trees, pattern, paradigms, vocab, n_sentences=10):
    arc_dir, context = pattern.split("\t")[:2]
    context = tuple(context.split("_"))
    l_values = pattern.split("\t")[2:]
    pattern_id = pattern.replace("\t", "!")

    ltm_paradigms = ltm_to_word(paradigms)

    output = []
    constr_id = 0

    n_vocab_unk = 0
    n_paradigms_unk = 0
    # 'nodes' constitute Y, without X or Z included
    for context, l, r, t, nodes in grep_morph_pattern(trees, context, l_values, arc_dir):
        #pos_constr = "_".join(n.pos for n in t.nodes[l.index - 1: r.index])

        # filter model sentences with unk and the choice word not in vocab
        if not all([n.word in vocab for n in nodes + [l, r]]):
            n_vocab_unk += 1
            continue
        if not is_good_form(r.word, r.word, r.morph, r.lemma, r.pos, vocab, ltm_paradigms):
            n_paradigms_unk += 1
            continue

        prefix = " ".join(n.word for n in t.nodes[:r.index])

        for i in range(n_sentences):
            # sent_id = 0 - original sentence with good lexical items, other sentences are generated
            if i == 0:
                new_context = " ".join(n.word for n in t.nodes)
                form = r.word
                form_alt = get_alt_form(r.lemma,r.pos,r.morph,ltm_paradigms)
                lemma = r.lemma
            else:
                new_context = generate_context(t.nodes, paradigms, vocab)
                random_forms = choose_random_forms(ltm_paradigms,vocab, r.pos,r.morph, n_samples=1, gold_word=r.word)
                if len(random_forms) > 0:
                    lemma, form, form_alt = random_forms[0]
                else:
                    # in rare cases, there is no (form, form_alt) both in vocab
                    # original form and its alternation are not found because e.g. one or the other is not in paradigms
                    # (they should anyway be in the vocabulary)
                    lemma, form = r.lemma, r.word
                    form_alt = get_alt_form(r.lemma, r.pos, r.morph, ltm_paradigms)

            # constr_id sent_id Z_index Z_pos Z_gold_morph
            gold_str = "\t".join([pattern_id, str(constr_id), str(i),
                                  str(r.index - 1), r.pos, r.morph, form, form_alt, lemma,
                                  str(l.index - 1), l.pos, prefix]) + "\n"

            output.append((new_context + " <eos>\n", gold_str))

        constr_id += 1

    print("Problematic sentences vocab/paradigms", n_vocab_unk, n_paradigms_unk)
    return output


def is_content_word(pos):
    return pos in ["ADJ", "NOUN", "VERB", "PROPN", "NUM", "ADV"]


def generate_context(nodes, paradigms, vocab):
    output = []

    for i in range(len(nodes)):
        substitutes = []
        n = nodes[i]
        # substituting content words
        if is_content_word(n.pos):
            for word in paradigms:
                if word == n.word:
                    continue
                # matching capitalization and vowel
                if not match_features(word, n.word):
                    continue

                tag_set = set([p[1] for p in paradigms[word]])
                # use words with unambiguous POS
                if len(tag_set) == 1 and tag_set.pop() == n.pos:
                    for _, _, morph, freq in paradigms[word]:
                        if n.morph == morph and int(freq) > 1 and word in vocab:
                            substitutes.append(word)
            if len(substitutes) == 0:
                output.append(n.word)
            else:
                output.append(random.choice(substitutes))
        else:
            output.append(n.word)
    return " ".join(output)


def choose_random_forms(ltm_paradigms, vocab, gold_pos, morph, n_samples=10, gold_word=None):
    candidates = set()

    #lemma_tag_pairs = ltm_paradigms.keys()
    #test_lemmas = [l for l, t in lemma_tag_pairs]

    for lemma in ltm_paradigms:
        poses = list(ltm_paradigms[lemma].keys())
        if len(set(poses)) == 1 and poses.pop() == gold_pos:
            form = ltm_paradigms[lemma][gold_pos][morph]
            _, morph_alt = alt_numeral_morph(morph)
            form_alt = ltm_paradigms[lemma][gold_pos][morph_alt]

            if not is_good_form(gold_word, form, morph, lemma, gold_pos, vocab, ltm_paradigms):
                continue

            candidates.add((lemma, form, form_alt))

    if len(candidates) > n_samples:
        return random.sample(candidates, n_samples)
    else:
        return random.sample(candidates, len(candidates))


def main():
    parser = argparse.ArgumentParser(description='Generating sentences based on patterns')

    parser.add_argument('--treebank', type=str, required=True,
                        help='input file (in a CONLL column format)')
    parser.add_argument('--paradigms', type=str, required=True, help="the dictionary of tokens and their morphological annotations")
    parser.add_argument('--vocab', type=str, required=True,help='(LM) Vocabulary to generate words from')
    parser.add_argument('--patterns', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="prefix for generated text and annotation data")
    parser.add_argument('--lm_data', type=str, required=False, help="path to LM data to estimate word frequencies")
    args = parser.parse_args()

    trees = tm.load_trees_from_conll(args.treebank)
    for t in trees:
        t.remerge_segmented_morphemes()

    paradigms = read_paradigms(args.paradigms)

    f_text = open(args.output + ".text", "w")
    f_gold = open(args.output + ".gold", "w")
    f_eval = open(args.output + ".eval", "w")

    output = []
    vocab = load_vocab(args.vocab)

    for line in open(args.patterns, "r"):
        print("Generating sentences with pattern", line.strip())
        #l_values = ('Gender=Fem|Number=Sing','Gender=Masc|Number=Plur')
        data = generate_morph_pattern_test(trees, line.strip(), paradigms, vocab)
        output.extend(data)
        print("Generated", len(data), "sentences")

    random.shuffle(output)
    sents, golds = zip(*output)

    f_text.writelines(sents)
    f_gold.writelines(golds)
    # save the index of the target word to evaluate
    f_eval.writelines([g.split("\t")[3] + "\n" for g in golds])

    ##############################################################
    # Make a readable data table with fields useful for analysis #
    ##############################################################

    data = transform_gold(golds)
    data = pd.DataFrame(data, columns=["pattern_id", "constr_id", "sent_id", "correct_number", "form", "class"])
    data.loc[data.sent_id == 0, "type"] = "original"
    data.loc[data.sent_id > 0, "type"] = "generated"

    # getting simpler pattern labels
    patterns = {p: "__".join(p.split("!")[:2]) for p in set(data.pattern_id)}
    data["pattern"] = data["pattern_id"].map(patterns)

    df_sents = extract_sent_features(sents, golds, vocab, paradigms)
    full_df = data.merge(df_sents, on=["pattern_id", "constr_id", "sent_id"])

    if args.lm_data:
        freq_dict = vocab_freqs(args.lm_data + "/train.txt", vocab)
        full_df["freq"] = full_df["form"].map(freq_dict)
        fields = ["pattern", "constr_id", "sent_id", "correct_number", "form", "class", "type", "prefix", "n_attr",
                  "punct","freq", "len_context", "len_prefix", "sent"]
    else:
        fields = ["pattern", "constr_id", "sent_id", "correct_number", "form", "class", "type", "prefix", "n_attr",
                  "punct","len_context", "len_prefix", "sent"]


    full_df[fields].to_csv(args.output + ".tab", sep="\t", index=False)


if __name__ == "__main__":
    main()