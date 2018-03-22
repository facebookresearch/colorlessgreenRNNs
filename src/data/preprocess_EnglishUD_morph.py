# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys

file_name = sys.argv[1]



for l in open(file_name):
    fields = l.strip().split("\t")
    if len(fields) == 10:
        morph = fields[5]
        # annotate non-singular verbs in present as Plural
        if "Tense=Pres" in morph and "VerbForm=Fin" in morph and "Number=Sing" not in morph:
            morph = morph + "|Number=Plur"
            s_m = morph.split("|")
            s_m.sort()
            morph = "|".join(s_m)
        elif "Number=Sing" in morph:
            feats = morph.split("|")
            # remove Person=3 annotation (since we don't have it for non-singular cases)
            feats = [f for f in feats if "Person=3" not in f]
            morph = "|".join(feats)
        print("\t".join(fields[:5] + [morph, ] + fields[6:]))
    else:
        print(l.strip())
