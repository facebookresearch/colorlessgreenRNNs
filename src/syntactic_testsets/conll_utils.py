# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
#!/usr/bin/env python


import sys
import re

from collections import namedtuple


ConllConfig = namedtuple('CONLL_config',
                         ['INDEX', 'WORD', 'POS', 'LEMMA', 'MORPH',
                          'HEAD_INDEX', 'DEP_LABEL',
                          'OFFSET', 'ROOT_INDEX', 'NCOLS'], verbose=False)

UD_CONLL_CONFIG = ConllConfig(INDEX=0, WORD=1, LEMMA=2, POS=3, MORPH=5,
                              HEAD_INDEX=6, DEP_LABEL=7, OFFSET=1, ROOT_INDEX=0, NCOLS=10)
UD_CONLL_FINE_POS_CONFIG = ConllConfig(INDEX=0, WORD=1, LEMMA=2, POS=4, MORPH=5,
                                       HEAD_INDEX=6, DEP_LABEL=7, OFFSET=1, ROOT_INDEX=0, NCOLS=10)
CONLL09_CONFIG = ConllConfig(INDEX=0, WORD=1, LEMMA=2, POS=4, MORPH=6,   #TODO check morph column id
                             HEAD_INDEX=8, DEP_LABEL=10, OFFSET=1, ROOT_INDEX=0, NCOLS=12)
ZGEN_CONLL_CONFIG = ConllConfig(INDEX=None, WORD=0, LEMMA=0, POS=1, MORPH=None,
                                HEAD_INDEX=2, DEP_LABEL=3, OFFSET=0, ROOT_INDEX=-1, NCOLS=4)
ARCS_CONLL_CONFIG = ConllConfig(INDEX=0, WORD=1, LEMMA=1, POS=2, MORPH=None,
                                HEAD_INDEX=3, DEP_LABEL=6, OFFSET=1, ROOT_INDEX=0, NCOLS=7)

DEP_LABEL_TYPES = {
    "core": "ccomp csubj csubjpass dobj iobj nsubj nsubjpass xcomp".split(),
    "non_core": """acl discourse nmod advcl dislocated nummod advmod expl parataxis amod foreign remnant appos
          goeswith reparandum compound list root -NONE- conj mwe vocative dep name""".split(),
    "func": "aux auxpass case cc cop det mark neg".split(),
    "other": "punct".split()}


def get_config(name):
    if name == "UD":
        return UD_CONLL_CONFIG
    elif name == "ZGEN":
        return ZGEN_CONLL_CONFIG
    elif name == "CONLL09":
        return CONLL09_CONFIG
    elif name == "UD_fine_pos":
        return UD_CONLL_FINE_POS_CONFIG


def read_blankline_block(stream):
    s = ''
    list = []
    while True:
        line = stream.readline()
        # End of file:
        if not line:
            list.append(s)
            return list
        # Blank line:
        elif line and not line.strip():
            list.append(s)
            s = ''
        # Other line:
        # in Google UD some lines can be commented and some can have multiword expressions/fused morphemes introduced by "^11-12    sss"
        #  and not re.match("[0-9]+-[0-9]+",line) and not line.startswith("<")
        elif not line.startswith("#"): # and "_\t_\t_\t_\t_\t" in line):
            #        and not line.startswith("<"):
            s += line


def read_sentences_from_columns(stream):
    # grids are sentences in column format
    grids = []
    for block in read_blankline_block(stream):
        block = block.strip()
        if not block: continue

        grid = [line.split('\t') for line in block.split('\n')]

        appendFlag = True
        # Check that the grid is consistent.
        for row in grid:
            if len(row) != len(grid[0]):
                print(grid)
                #raise ValueError('Inconsistent number of columns:\n%s'% block)
                sys.stderr.write('Inconsistent number of columns', block)
                appendFlag = False
                break

        if appendFlag:
            grids.append(grid)

    return grids


def output_conll(sentences, prefix):
    f_gold = open(prefix + "_conll.gold", "w")
    f_guess = open(prefix + "_conll.guess", "w")
    for sentence in sentences:
        for (num, word, pos, correct_dep, guess_dep) in sentence:
            f_gold.write("\t".join([num, word, word, pos, pos, "_", correct_dep, "_", "_", "_"]) + "\n")
            f_guess.write("\t".join([num, word, word, pos, pos, "_", guess_dep, "_", "_", "_"]) + "\n")
        f_gold.write("\n")
        f_guess.write("\n")


def pprint(column_sentence):
    for row in column_sentence:
        print("\t".join([word for word in row]))
    print("")


def write_conll(sentences, file_out):
    for sentence in sentences:
        #    print "\n".join("\t".join(word for word in row)for row in sentence)
        file_out.write("\n".join("\t".join(word for word in row) for row in sentence))
        file_out.write("\n\n")


def pseudo_rand_split(sentences):
    i = 0
    train = []
    test = []
    for sentence in sentences:
        i += 1
        if i < 10:
            train.append(sentence)
        else:
            test.append(sentence)
            i = 0
    return train, test

'''

def main():
  s = conll_utils()
  s.read()
  s.output_short_sentences()
#s.print_dep_length()


class conll_utils(object):


      #      num_sents[len(extract_arcs(tree))] += 1

#    print "\n".join("%d\t%f\t%f" % (size, counts_real[size]/float(num_sents[size]), counts_rand[size]/float(num_sents[size])) for size in counts_real.keys())

#print dep_length(extract_arcs(tree))



  def output_short_sentences(self):
    sentences = read_sentences_from_columns(open(self.input))
    for sentence in sentences:
        if len(sentence) == 10: #  and len(sentence) > 8:
          pprint(sentence)
        
    
     

  def read(self):
    #sys.stderr.write("Main..\n")
    self.sentences = read_sentences_from_columns(open(self.input))
    #print self.sentences
    
    """ self.correct_trees = []
    self.guess_trees = []
    for sentence in self.sentences:
        self.correct_trees.append([row[:4] for row in sentence])
        self.guess_trees.append([row[:3] + [row[4]] for row in sentence]) """

    
  def __init__(self):
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--num_training_iterations", dest="iterations", default=5, type="int", help="Number of training iterations")
    optparser.add_option("-N", "--num_training_sentences", dest="num_sents", default=1000, type="int", help="Number of training sentences to use")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Score threshold for alignment")
    optparser.add_option("-d", "--display_count", dest="display_count", default=5, type="int", help="Number of alignments to display")    
    optparser.add_option("-i", "--input", dest="input", default="test", help="Input file name")    
    optparser.add_option("-e", "--evaluation", dest="evaluation", default="undirected", help="Type of dependency evaluation")    
    (opts, args) = optparser.parse_args()
    self.input = opts.input
    self.evaluation = opts.evaluation
    return

  def accuracy():

    
    sum = 0
    length = 0
    print len(self.sentences)
    if (self.evaluation == "directed"):
        for sentence in self.sentences:
            sum += correct_dir(sentence)
            length += len(sentence)
    #undirected
    else:
        for (correct_tree, guess_tree) in zip(self.correct_trees, self.guess_trees):
            sum += correct_undir(correct_tree, guess_tree)
            print "\n".join(str(row) for row in zip(correct_tree, guess_tree))
            print correct_undir(correct_tree, guess_tree)
        
            length += len(correct_tree)

    print sum / float(length)
    
    for c1, c2 in zip(collect_statistics(self.correct_trees), collect_statistics(self.guess_trees)):
        print ' '.join(str(i) for i in c1[0]) + "\t" + str(c1[1])

    #output_conll(self.sentences, self.input)



if __name__ == "__main__":
    main()
'''

