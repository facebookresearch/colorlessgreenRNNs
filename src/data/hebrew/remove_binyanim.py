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
    feats = morph.split("|")
    feats = [f for f in feats if "HebBi" not in f and "HebCo" not in f and "Voice" not in f]
    morph = "|".join(feats)
    print("\t".join(fields[:5] + [morph,] + fields[6:]))
  else:
    print(l.strip())
