# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import subprocess


def query_KenLM(lm_file, file_name, kenlm_path="/private/home/gulordava/kenlm/build/bin/"):
    """
    :param lm_file: language model
    :param file_name: file with (partial) sentences to test
    :return: a list of probabilities of the last word of each sentence
    """

    command = kenlm_path + "query " + lm_file + ' < ' + file_name + " -n"
    KenLM_query = subprocess.getstatusoutput(command)[1]

    lines = KenLM_query.split("\n")
    skip = ["This binary file contains probing hash tables.",
            "Loading the LM will be faster if you build a binary file."]
    if any(s in lines[0] for s in skip):
        lines = lines[1:]

    result_probs = []
    for line in lines:
        # last ngram is Total + OOV
        try:
            result_probs.append(float(line.split('\t')[-2].split(" ")[2]))
        except (IndexError, ValueError) as e:
            print(line)

    return result_probs, lines