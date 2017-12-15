
lang=$1
model_type=$2

path="/private/home/gulordava/lstm_hyperparameters_exps/$lang/$model_type"
path_lm_data="/private/home/gulordava/colorlessgreen/data/lm/$lang"
test_file=$path_lm_data/valid.txt
output_path=$path/valid_ppls/
#test_file=/private/home/gulordava/edouard_data/itwiki/it-ud-train.txt_eos
#test_file=/private/home/gulordava/edouard_data/hewiki/hebrew.conllu.txt_eos
#output_path=$path/valid_ppls_treebank/

for f in $path/models/* ; do
    m=`basename $f`
    m="${m%.*}"
    echo $m

    EXP_NAME=eval_ppl/$m/
    mkdir -p /checkpoint/$USER/jobs/$EXP_NAME
    mkdir -p $output_path

    sbatch --job-name=$EXP_NAME \
    --output=/checkpoint/%u/jobs/$EXP_NAME/%j.out \
    --error=/checkpoint/%u/jobs/$EXP_NAME/%j.err \
    --partition=learnfair --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:1 \
    --time=00:20:00 \
    --wrap="python ~/lm_code/evaluate_test_perplexity.py --data $path_lm_data --checkpoint $f --test $test_file --cuda > $output_path/${m}.txt"
done

