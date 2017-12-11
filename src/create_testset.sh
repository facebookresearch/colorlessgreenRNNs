
# path to treebank
# should be preprocessed for English and Hebrew
treebank="$HOME/edouard_data/enwiki/en-ud-train.conllu_new"
lang="English"

# default paths that should work for all languages
lm_data="../data/lm/$lang/"

mkdir -p tmp/$lang

# extracting word forms, lemmas and morphological features
echo "== collect paradigms =="
python data/collect_paradigms.py --input $treebank --output tmp/$lang/paradigms.txt --min_freq 0

echo "== extract agreement constructions =="
# extracting patterns which correpond to agreement relation
python syntactic_testsets/extract_dependency_patterns.py --treebank $treebank --output tmp/$lang/ --features Number --vocab $lm_data/vocab.txt --paradigms tmp/$lang/paradigms.txt

echo "== generate test set =="
# given a list of patterns, grep corresponding instances which satisfy generation conditions 
python syntactic_testsets/generate_nonsense.py --treebank $treebank --paradigms tmp/$lang/paradigms.txt \
                                               --vocab $lm_data/vocab.txt --patterns tmp/$lang/patterns.txt --output ../data/agreement/$lang/generated


