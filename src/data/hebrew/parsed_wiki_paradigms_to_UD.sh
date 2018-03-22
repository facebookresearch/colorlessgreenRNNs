#!/usr/bin/env bash
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

sed "s/num=P/Number=Plur/" paradigms_100M_freq10.txt | sed s/num=S/Number=Sing/ | sed s/gen=F/Gender=Fem/ | sed s/gen=M/Gender=Masc/ | sed "s/NNT	/NN	Definite=Cons|/" | sed s/NNP/PROPN/ | sed s/NN/NOUN/ | sed s/VB/VERB/ | sed s/JJT/ADJ/ | sed s/JJ/ADJ/ | sed s/IN/ADP/ | sed s/per/Person/ | sed s/tense=PAST/Tense=Past/ | sed s/tense=FUTURE/Tense=Fut/ | sed s/tense=IMPERATIVE/Mood=Imp/| sed s/Person=A/Person=1,2,3/ | sed s/Gender=Fem\|Gender=Masc/Gender=Fem,Masc/ | sed s/tense=BEADPONI/VerbForm=Part/ 
