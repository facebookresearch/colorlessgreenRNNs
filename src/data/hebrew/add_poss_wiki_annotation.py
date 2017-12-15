import sys
file_name = sys.argv[1]

for l in open(file_name):
  fields = l.strip().split("\t")
  if len(fields) == 10:
    morph = fields[5]
    fine_tag = fields[4]
    if "NN+POS+PRP" in fine_tag:
       morph = morph + "|Poss=Yes"
    print("\t".join(fields[:5] + [morph,] + fields[6:]))
  else:
    print(l.strip())
