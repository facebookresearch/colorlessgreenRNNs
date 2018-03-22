# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""

Classes Node, Arc, DependencyTree providing functionality for syntactic dependency trees

"""

from __future__ import print_function, division

import re
from queue import Queue

import conll_utils as conll_utils


class Node(object):

    def __init__(self, index=None, word="", lemma="", head_id=None, pos="", dep_label="", morph="_",
                 size=None, dep_label_new=None):
        """
        :param index: int
        :param word: str
        :param head_id: int
        :param pos: str
        :param dep_label: str
        """
        self.index = index
        self.word = word
        self.lemma = lemma
        self.head_id = head_id
        self.pos = pos
        self.dep_label = dep_label
        self.morph = morph
        if dep_label_new is None:
            self.dep_label_new = dep_label
        else:
            self.dep_label_new = dep_label_new
        # to assign after tree creation
        self.size = size
        self.dir = None

    def __str__(self):
        return "\t".join([str(self.index), self.word, self.pos, self.morph, str(self.head_id), str(self.dep_label)])

    def __repr__(self):
        return "\t".join([str(v) for (a, v) in self.__dict__.items() if v])

    @classmethod
    def from_str(cls, string):
        index, word, pos, head_id, dep_label = [None if x == "None" else x for x in string.split("\t")]
        return Node(index, word, head_id, pos, dep_label)

    def __eq__(self, other):
        return other is not None and \
               self.index == other.index and \
               self.word == other.word and \
               self.head_id == other.head_id and \
               self.pos == other.pos and \
               self.dep_label == other.dep_label

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

    def is_root(self):
        generic_root = DependencyTree.generic_root(conll_utils.UD_CONLL_CONFIG)
        if self.word == generic_root.word and self.pos == generic_root.pos:
            return True
        return False


class Arc(object):
    LEFT = "L"
    RIGHT = "R"

    def __init__(self, head, direction, child):
        self.head = head
        self.dir = direction
        self.child = child
        self.dep_label = child.dep_label

    def __str__(self):
        return str(self.head) + " " + self.dir + " " + str(self.child)

    def __repr__(self):
        return str(self)

    @classmethod
    def from_str(cls, string):
        head_str, dir, child_str = string.split(" ")
        return Arc(Node.from_str(head_str), dir, Node.from_str(child_str))

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

    def length(self):
        # arcs to ROOT node have length 0
        if self.head.is_root():
            return 0
        else:
            return abs(self.child.index - self.head.index)


class DependencyTree(object):
    def __init__(self, nodes, arcs, config, fused_nodes):
        self.nodes = nodes
        self.arcs = arcs
        self.assign_sizes_to_nodes()
        self.config = config
        # for UD annotation to be able to recover original sentence (without split morphemes)
        self.fused_nodes = fused_nodes

    def __str__(self):
        return "\n".join([str(n) for n in self.nodes])

    def __repr__(self):
        return str(self)

    def children(self, head):
        children = []
        for arc in self.arcs:
            if arc.head == head:
                children.append(arc.child)
        return children

    def assign_sizes_to_nodes(self):
        for node in self.nodes:
            node.size = len(self.children(node)) + 1

    def reindex(self, nodes, conll_config):
        """ After reordering 'nodes' list reflects the final order of nodes, however the indices of node objects
          do not correspond to this order. This function fixes it. """
        new_positions = {}
        new_nodes = [] # in order
        for i in range(len(nodes)):
            new_positions[nodes[i].index] = i

        for i in range(len(nodes)):
            new_nodes.append(nodes[i])
            if nodes[i].head_id == conll_config.ROOT_INDEX:
                nodes[i].index = i + conll_config.OFFSET
            else:
                nodes[i].index = i + conll_config.OFFSET
                nodes[i].head_id = new_positions[nodes[i].head_id] + conll_config.OFFSET
        self.nodes = new_nodes

    def remove_node(self, node_x):
        assert len(self.children(node_x)) == 0
        self.nodes.remove(node_x)
        for node in self.nodes:
            if node.head_id > node_x.index:
                node.head_id = node.head_id - 1
            if node.index > node_x.index:
                node.index = node.index - 1

        for i in range(len(self.fused_nodes)):
            start, end, token = self.fused_nodes[i]
            if start > node_x.index:
                start = start - 1
            if end > node_x.index:
                end = end - 1
            self.fused_nodes[i] = (start, end, token)


    def subtree(self, head):
        elements = set()
        queue = Queue()
        queue.put(head)
        #head_ = Node(head.index, head.word, head.pos + "X")
        elements.add(head)
        visited = set()
        while not queue.empty():
            next_node = queue.get()
            if next_node in visited:
                continue
            visited.add(next_node)
            for child in self.children(next_node):
                elements.add(child)
                queue.put(child)

        return sorted(elements, key=lambda element: int(element.index))

    def is_projective_arc(self, arc):
        st = self.subtree(arc.head)
        # all nodes in subtree of the arc head
        st_idx = [node.index for node in st]
        # span between the child and the head
        indexes = range(arc.child.index + 1, arc.head.index) if arc.child.index < arc.head.index else range(
            arc.head.index + 1, arc.child.index)
        # each node/word between child and head should be part of the subtree
        # if not, than the child-head arc is crossed by some other arc and is non-projective
        for i in indexes:
            if i not in st_idx:
                return False
        return True

    def is_projective(self):
        return all(self.is_projective_arc(arc) for arc in self.arcs)

    def length(self):
        return sum(arc.length() for arc in self.arcs)

    def average_branching_factor(self):
        heads = [node.head_id for node in self.nodes]
        return len(self.nodes)/len(set(heads))

    def root(self):
        return DependencyTree.generic_root(self.config)

    def remerge_segmented_morphemes(self):
        """
        UD format only: Remove segmented words and morphemes and substitute them by the original word form
        - all children of the segments are attached to the merged word form
        - word form features are assigned heuristically (should work for Italian, not sure about other languages)
            - pos tag and morphology (zero?) comes from the first morpheme
        :return:
        """
        for start, end, token in self.fused_nodes:
            # assert start + 1 == end, t
            self.nodes[start - 1].word = token

            for i in range(end - start):
                # print(i)
                if len(self.children(self.nodes[start])) != 0:
                    for c in self.children(self.nodes[start]):
                        c.head_id = self.nodes[start - 1].index
                        self.arcs.remove(Arc(child=c, head=self.nodes[start], direction=c.dir))
                        self.arcs.append(Arc(child=c, head=self.nodes[start - 1], direction=c.dir))
                assert len(self.children(self.nodes[start])) == 0, (self, start, end, token, i, self.arcs)
                self.remove_node(self.nodes[start])
                # print(t)
                #        print(t)
        self.fused_nodes = []

    @classmethod
    def generic_root(cls, conll_config):
        return Node(conll_config.ROOT_INDEX, "ROOT", "ROOT", 0, "ROOT", size=0)

    @classmethod
    def from_sentence(cls, sentence, conll_config):
        nodes = []
        fused_nodes = []

        for i in range(len(sentence)):
            row = sentence[i]

            if conll_config.MORPH is not None:
                morph = row[conll_config.MORPH]
            else:
                morph = "_"

            # saving original word segments separated in UD (e.g. Italian darglielo -> dare + gli + lo)
            if conll_config == conll_utils.UD_CONLL_CONFIG:
                if re.match(r"[0-9]+-[0-9]+", row[0]):
                    fused_nodes.append((int(row[0].split("-")[0]), int(row[0].split("-")[1]), row[1]))
                    continue
                # empty elements (e.g. copula in Russian)
                if re.match(r"[0-9]+\.[0-9]+", row[0]):
                    continue

            if conll_config.INDEX is not None:
                nodes.append(
                    Node(int(row[conll_config.INDEX]),
                         row[conll_config.WORD],
                         row[conll_config.LEMMA],
                         int(row[conll_config.HEAD_INDEX]),
                         pos=row[conll_config.POS],
                         dep_label=row[conll_config.DEP_LABEL],
                         morph=morph))
            else:
                nodes.append(Node(i,
                                  row[conll_config.WORD],
                                  row[conll_config.LEMMA],
                                  int(row[conll_config.HEAD_INDEX]),
                                  pos=row[conll_config.POS],
                                  dep_label=row[conll_config.DEP_LABEL],
                                  morph=morph))

        arcs = []
        for node in nodes:
            head_index = int(node.head_id)
            head_element = nodes[head_index - conll_config.OFFSET]
            if head_index == conll_config.ROOT_INDEX:
                arcs.append(Arc(cls.generic_root(conll_config), Arc.LEFT, node))
            elif head_index < int(node.index):
                arcs.append(Arc(head_element, Arc.RIGHT, node))
                node.dir = Arc.RIGHT
            else:
                arcs.append(Arc(head_element, Arc.LEFT, node))
                node.dir = Arc.LEFT
        return cls(nodes, arcs, conll_config, fused_nodes)

    def pprint(self, conll_config, lower_case=False):
        # TODO: change the indices of heads in accordance with the config
        s = ""
        for node in self.nodes:
            row = ["_"] * conll_config.NCOLS
            if conll_config.INDEX is not None:
                row[conll_config.INDEX] = str(node.index)
            if node.word:
                if lower_case:
                    row[conll_config.WORD] = node.word.lower()
                else:
                    row[conll_config.WORD] = node.word
            if node.pos:
                row[conll_config.POS] = node.pos
            if node.morph:
                row[conll_config.MORPH] = node.morph
            if node.lemma:
                row[conll_config.LEMMA] = node.lemma
            row[conll_config.HEAD_INDEX] = str(node.head_id)
            if node.dep_label:
                row[conll_config.DEP_LABEL] = node.dep_label
            s = s + "\t".join(row) + "\n"
        return s #.encode("utf-8")


def load_trees_from_conll(file_name, config=None):
    sentences = conll_utils.read_sentences_from_columns(open(file_name))
    # config for the default cases, to facilitate handling of multiple formats at the same time
    # for guaranteed performance, config should be supplied
    if config is None:
        if len(sentences[0][0]) == conll_utils.ZGEN_CONLL_CONFIG.NCOLS:
            config = conll_utils.ZGEN_CONLL_CONFIG
        elif len(sentences[0][0]) == conll_utils.UD_CONLL_CONFIG.NCOLS:
            config = conll_utils.UD_CONLL_CONFIG
        else:
            print("Unrecognised format of ", file_name)
            return None
    trees = []
    for s in sentences:
        trees.append(DependencyTree.from_sentence(s, config))
    return trees
