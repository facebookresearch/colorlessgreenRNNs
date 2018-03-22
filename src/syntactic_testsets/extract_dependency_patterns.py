# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import tree_module as tm

import argparse
import itertools
from collections import defaultdict

import numpy as np

from generate_utils import is_good_form
from utils import load_vocab, ltm_to_word, read_paradigms


def safe_log(x):
    np.seterr(divide='ignore', invalid='ignore')
    return np.where(x > 0.0001, np.log(x), 0.0)


def cond_entropy(xy):

    # normalise
    xy = xy / np.sum(xy)

    x_ = np.sum(xy, axis=1)
    y_ = np.sum(xy, axis=0)

    x_y = xy / y_
    # print(x_y)
    y_x = xy / x_.reshape(x_.shape[0], 1)
    # print(y_x)

    # Entropies: H(x|y) H(y|x) H(x) H(y)
    return np.sum(-xy * safe_log(x_y)), np.sum(-xy * safe_log(y_x)), np.sum(-x_ * safe_log(x_)), np.sum(
        -y_ * safe_log(y_))



def pos_structure(nodes, arc):
    """ Get a sequence of pos tags for nodes which are direct children of the arc head or the arc child
        nodes - the list of nodes of the context Y, between the head and the child (X, Z) of the arc
    """
    return tuple([n.pos for n in nodes if n.head_id in [arc.head.index, arc.child.index]])


def inside(tree, a):
    if a.child.index < a.head.index:
        nodes = tree.nodes[a.child.index: a.head.index - 1]
        l = a.child
        r = a.head
    else:
        nodes = tree.nodes[a.head.index: a.child.index - 1]
        l = a.head
        r = a.child
    return nodes, l, r


def features(morph, feature_list):
    #Definite=Def|Gender=Masc|Number=Sing|PronType=Art Tense=Past|VerbForm=Part
    if not feature_list:
        return morph

    all_feats = morph.split("|")
    feat_values = tuple(f for f in all_feats if f.split("=")[0] in feature_list)

    return "|".join(feat_values)


def morph_contexts_frequencies(trees, feature_list):
    """
    Collect frequencies for X Y Z tuples, where Y is a context defined by its surface structure
    and X and Z are connected by a dependency

    :param trees: dependency trees
    :return: two dictionaries for left and right dependencies
    """
    d_left = defaultdict(lambda: defaultdict(int))
    d_right = defaultdict(lambda: defaultdict(int))
    for t in trees:
        for a in t.arcs:
            if 3 < a.length() < 15 and t.is_projective_arc(a):
                # print("\n".join(str(n) for n in t.nodes))
                nodes, l, r = inside(t, a)
                substring = (l.pos,) + pos_structure(nodes, a) + (r.pos,)
                # print(substring)
                if substring:
                    if features(l.morph, feature_list) == "" or features(r.morph, feature_list) == "":
                        continue
                    #substring = substring + (a.dep_label,)
                    if a.dir == tm.Arc.LEFT:
                        d_left[substring][(features(l.morph, feature_list), features(r.morph, feature_list))] += 1
                    if a.dir == tm.Arc.RIGHT:
                        d_right[substring][(features(l.morph, feature_list), features(r.morph, feature_list))] += 1
    return d_left, d_right


def find_good_patterns(context_dict, freq_threshold):
    """
    :param context_dict: is a dictionary of type { Y context : {(X, Z) : count} }
                         for X Y Z sequences where X and Z could be of any type (tags, morph)

    :param freq_threshold: for filtering out too infrequent patterns
    :return: list of patterns - tuples (context, left1, left2) == (Y, X1, X2)
             (where X1 and X2 occur with different Zs)
    """
    patterns = []
    for context in context_dict:
        left_right_pairs = context_dict[context].keys()
        if len(left_right_pairs) == 0:
            continue
        left, right = zip(*left_right_pairs)

        left_v = set(left)

        d = context_dict[context]
        if len(left_v) < 2:
            continue

        for l1, l2 in itertools.combinations(left_v, 2):
            right_v = [r for (l, r) in left_right_pairs if l in (l1, l2)]
            if len(right_v) < 2:
                continue

            a = np.zeros((2, len(right_v)))
            for i, x in enumerate((l1, l2)):
                for j, y in enumerate(right_v):
                    a[(i, j)] = d[(x, y)]

            l_r, r_l, l_e, r_e = cond_entropy(a)
            mi = l_e - l_r

            count_l1 = 0
            count_l2 = 0
            for l, r in d:
                if l == l1:
                    count_l1 += d[(l, r)]
                if l == l2:
                    count_l2 += d[(l, r)]

            #print(l_r, r_l, l_e, r_e, mi)
            if l_r < 0.001 and count_l1 > freq_threshold and count_l2 > freq_threshold:
                patterns.append((context, l1, l2))
                print(context, l_r, mi)
                print(l1, l2, count_l1, count_l2)
                #for l, r in d:
                #    if l in (l1, l2) and d[(l, r)] > 0 :
                #        print(l, r, d[(l, r)])
    return patterns


def grep_morph_pattern(trees, context, l_values, dep_dir, feature_list=None):
    """
    :param context: Y
    :param l_values:  l_values are relevant X values
    :param dep_dir:
    :return: generator of (context-type, l, r, tree, Y nodes) tuples
    """
    if feature_list is None:
        feature_list = ['Number']

    for t in trees:
        for a in t.arcs:
            if 3 < a.length() < 15 and t.is_projective_arc(a):
                if a.child.pos == "PUNCT" or a.head.pos == "PUNCT":
                    continue
                #print("\n".join(str(n) for n in t.nodes))
                nodes, l, r = inside(t, a)

                if a.dir != dep_dir:
                    continue

                if not any(m in l.morph for m in l_values):
                    #print(features(l.morph), l_values)
                    continue
                if features(r.morph, feature_list) != features(l.morph, feature_list):
                    continue
                substring = (l.pos,) + pos_structure(nodes, a) + (r.pos,)

                if substring == context:
                    #print(substring, context)
                    yield context, l, r, t, nodes


def main():
    parser = argparse.ArgumentParser(
        description='Extracting dependency-based long-distance agreement patterns')

    parser.add_argument('--treebank', type=str, required=True,
                        help='Path of the input treebank file (in a column format)')
    parser.add_argument('--output', type=str, required=True,
                        help="Path for the output files")
    parser.add_argument('--features', type=str, default="Number",
                        help="A list of morphological features which will be used, in Number|Case|Gender format")
    parser.add_argument('--freq', type=int, default=5, help="minimal frequency")
    parser.add_argument('--vocab', type=str, required=False, help="LM vocab - to compute which sentences have OOV")
    parser.add_argument('--paradigms', type=str, required=False, help="File with morphological paradigms - to compute"
                                                                      "which sentences have both target pairs")

    args = parser.parse_args()

    if args.vocab:
        vocab = load_vocab(args.vocab)
    else:
        vocab = []

    print("Loading trees")
    trees = tm.load_trees_from_conll(args.treebank)

    # needed for original UD treebanks (e.g. Italian) which contain spans, e.g. 10-12
    # annotating mutlimorphemic words as several nodes in the tree
    for t in trees:
        t.remerge_segmented_morphemes()

    if args.features:
        args.features = args.features.split("|")
        print("Features", args.features)


    print("Extracting contexts")
    context_left_deps, context_right_deps = morph_contexts_frequencies(trees, args.features)


    # filtering very infrequent cases
    filter_threshold = 1
    context_left_deps = defaultdict(lambda: defaultdict(int), {c: defaultdict(int,
            {lr: freq for lr, freq in d.items() if freq > filter_threshold}) for c, d in context_left_deps.items()})
    context_right_deps = defaultdict(lambda: defaultdict(int), {c: defaultdict(int,
            {lr: freq for lr, freq in d.items() if freq > filter_threshold}) for c, d in context_right_deps.items()})


    print("Finding good patterns")
    good_patterns_left = find_good_patterns(context_left_deps, args.freq)
    good_patterns_right = find_good_patterns(context_right_deps, args.freq)

    f_out = open(args.output + "/patterns.txt", "w")

    print("Saving patterns and sentences matching them")

    ltm_paradigms = ltm_to_word(read_paradigms(args.paradigms))

    for p in good_patterns_left:
        f_out.write("L\t" + "_".join(x for x in p[0]) + "\t" + "\t".join(p[1:]) + "\n")
        print("L\t" + "_".join(x for x in p[0]) + "\t" + "\t".join(p[1:]) + "\n")

        f_out_grep = open(args.output + "/L_" + "_".join(x for x in p[0]), "w")
        for context, l, r, t, nodes in grep_morph_pattern(trees, p[0], p[1:], tm.Arc.LEFT, args.features):
            #print(l.morph + " " + r.morph + "\t" + l.word + " " + " ".join([n.word for n in nodes]) + " " + r.word)

            in_vocab = all([n.word in vocab for n in nodes + [l, r]])
            in_paradigms = is_good_form(r.word, r.word, r.morph, r.lemma, r.pos, vocab, ltm_paradigms)
            f_out_grep.write(features(l.morph, args.features) + " " + features(r.morph, args.features) +
                             "\t" + str(in_vocab) + str(in_paradigms) + "\t" + l.word + " " + " ".join([n.word for n in nodes]) + " " + r.word + "\n")
        f_out_grep.close()

    for p in good_patterns_right:
        f_out.write("R\t" + "_".join(x for x in p[0]) + "\t" + "\t".join(p[1:]) + "\n")
        print("R\t" + "_".join(x for x in p[0]) + "\t" + "\t".join(p[1:]) + "\n")

        f_out_grep = open(args.output + "/R_" + "_".join(x for x in p[0]), "w")
        for context, l, r, t, nodes in grep_morph_pattern(trees, p[0], p[1:], tm.Arc.RIGHT, args.features):
            #print(l.morph + " " + r.morph + "\t" + l.word + " " + " ".join([n.word for n in nodes]) + " " + r.word)
            in_vocab = all([n.word in vocab for n in nodes + [l, r]])
            in_paradigms = is_good_form(r.word, r.word, r.morph, r.lemma, r.pos, vocab, ltm_paradigms)
            f_out_grep.write(features(l.morph, args.features)+ " " + features(r.morph, args.features) +
                             "\t" + str(in_vocab) + str(in_paradigms) + "\t" + l.word + " " + " ".join([n.word for n in nodes]) + " " + r.word + "\n")
        f_out_grep.close()

    f_out.close()


if __name__ == "__main__":
    main()