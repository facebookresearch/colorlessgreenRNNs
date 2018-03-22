# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import conll_utils
import tree_module as tm


def remove_segmented_morphemes_hebrew(t):
    for start, end, token in t.fused_nodes:
        # assert start + 1 == end, t

        # don't need to change anything
        if all(not n.word.startswith("_") and not n.word.endswith("_") for n in t.nodes[start - 1:end]):
            # print(start, end, token)
            continue

        tokens_separated = ""
        for n in t.nodes[start - 1:end]:
            if not n.word.startswith("_") and not n.word.endswith("_"):
                start = start + 1
                tokens_separated = tokens_separated + n.word
            else:
                break

        # print("tokens sep", tokens_separated)

        head = None
        for n in t.nodes[start - 1:end]:
            # print(start-1, end-1)
            # print(n.head_id)
            if n.head_id > end or n.head_id < start:
                # in two sentences two parts of a word had two different heads
                # in 20 cases several parts of a word had the same head - annotated with 'fixed' dependency
                # assert head is None, (t, t.fused_nodes, start, end, t.nodes[start])
                # if head is not None and head.head_id == n.head_id:
                #    print("fixed")

                head = n
        assert head is not None, (t, t.fused_nodes, start, end, t.nodes[start])

        # print(start - 1, end)

        # print("head", head)
        merged_part = token[len(tokens_separated):]
        # print("merged part", )
        if merged_part == "":
            start = start - 1
        else:
            t.nodes[start - 1].word = token[len(tokens_separated):]
            t.nodes[start - 1].lemma = head.lemma
            t.nodes[start - 1].pos = head.pos

        t.nodes[start - 1].morph = head.morph
        t.nodes[start - 1].dep_label = head.dep_label

        # print(t.nodes[start - 1].head_id)


        for i in range(end - start):
            if t.nodes[start].dep_label == "nmod:poss":
                t.nodes[start - 1].morph = t.nodes[start - 1].morph + "|Poss=Yes"
            # print(i)
            if len(t.children(t.nodes[start])) != 0:
                for c in t.children(t.nodes[start]):
                    c.head_id = t.nodes[start - 1].index
                    t.arcs.remove(tm.Arc(child=c, head=t.nodes[start], direction=c.dir))
                    t.arcs.append(tm.Arc(child=c, head=t.nodes[start - 1], direction=c.dir))
            assert len(t.children(t.nodes[start])) == 0, (t, start, end, token, i, t.arcs)
            t.remove_node(t.nodes[start])
            # print(t)

        # important, after removal of other nodes so that their dependencies get attached first to the right head
        t.nodes[start - 1].head_id = head.head_id

    t.fused_nodes = []


path = "/private/home/gulordava/edouard_data/hewiki/hebrew.conllu"
trees = tm.load_trees_from_conll(path)

for i, t in enumerate(trees):
    # in place
    remove_segmented_morphemes_hebrew(t)

f_trees_new = open(path + "_new", "w")
for t in trees:
    f_trees_new.write(t.pprint(conll_utils.UD_CONLL_CONFIG) + "\n")
    #print(t.pprint(conll_utils.UD_CONLL_CONFIG))