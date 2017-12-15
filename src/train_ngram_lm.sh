

#data=/private/home/gulordava/edouard_data/itwiki/50K_vocab_shuffled/
#path=/private/home/gulordava/ngram_lms/Italian/ 
#prefix=it

data=$1
path=$2
prefix=$3

mkdir -p $path

# build KN 5 gram model
if [ ! -f $path/${prefix}_lm.gz ] ; then
  $HOME/irstlm/scripts/build-lm.sh -i $data/train.txt -o $path/${prefix}_lm.gz -n 5 -s kneser-ney
fi

# arpa text file
if [ ! -f $path/${prefix}_lm.lm ] ; then
  $HOME/irstlm/bin/compile-lm $path/${prefix}_lm.gz $path/${prefix}_lm.lm --text=yes
fi

# fixing <s> problem (added to the ngram table, but inconsistently)
if [ ! -f ${path}/${prefix}_lm.lm_no_s ] ; then
  grep -v "<s>" $path/${prefix}_lm.lm > ${path}/${prefix}_lm.lm_no_s
fi

# TODO manually: decrease the 2,3,4,5grams counts (-1) 
# vim it_model_wo_s.lm

if [ ! -f $path/${prefix}.binary ] ; then
  $HOME/kenlm/build/bin/build_binary -s -i $path/${prefix}_lm.lm_no_s $path/${prefix}.binary
fi

#echo "il suo mondo" | $HOME/kenlm/build/bin/query $path/${prefix}.binary -n

$HOME/kenlm/build/bin/query -n -v summary $path/${prefix}.binary < $data/valid.txt
