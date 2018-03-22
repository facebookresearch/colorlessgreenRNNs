# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#



def is_vowel(c):
    return c in ["a","o","u","e","i","A","O","U","E","I","è"]


def alt_numeral_morph(morph):
    if "Number=Plur" in morph:
        morph_alt = morph.replace("Plur", "Sing")
        return "plur", morph_alt
    elif "Number=Sing" in morph:
        morph_alt = morph.replace("Sing", "Plur")
        return "sing", morph_alt


def is_good_form(gold_form, new_form, gold_morph, lemma, pos, vocab, ltm_paradigms):
    _, alt_morph = alt_numeral_morph(gold_morph)
    if not new_form in vocab:
        return False
    alt_form = ltm_paradigms[lemma][pos][alt_morph]
    if not alt_form in vocab:
        return False
    if gold_form is None:
        print(gold_form, gold_morph)
        return True
    if not match_features(new_form, gold_form):
        return False
    if not match_features(alt_form, gold_form):
        return False
    return True


def get_alt_form(lemma, pos, morph, ltm_paradigms):
    _, alt_morph = alt_numeral_morph(morph)
    return ltm_paradigms[lemma][pos][alt_morph]


def match_features(w1, w2):
    return w1[0].isupper() == w2[0].isupper() and is_vowel(w1[0]) == is_vowel(w2[0])

