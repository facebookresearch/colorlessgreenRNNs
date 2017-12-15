#input path_lm_data path_models path_test_data

lang=$1
models_path="/private/home/gulordava/lstm_hyperparameters_exps/$lang/ngram_lstm/models"
path_lm_data="/private/home/gulordava/colorlessgreen/data/lm/"

path_test_data="/private/home/gulordava/colorlessgreen/data/agreement/$lang/generated"

#l=`squeue -u gulordava --format "%R" -h`
#l=`echo $l | tr ' ' ','`

for f in $models_path/*hidden650*; do
    m=`basename $f`
    m="${m%.*}"
    echo $m

    EXP_NAME=eval_ngram_lstm/$lang/$m

    mkdir -p /checkpoint/$USER/jobs/$EXP_NAME
    
    #while true
    #do
    #  l=`squeue -u gulordava --format "%R" -h`
    #  l=`echo $l | tr ' ' ','`
    #  echo $l
    #  if [[ ! $l == *"(None)"* ]]; then
    #    break
    #  fi
    #  sleep 3
    #done 

    #--exclude $l

    sbatch --job-name=$EXP_NAME \
    --output=/checkpoint/%u/jobs/$EXP_NAME/%j.out \
    --error=/checkpoint/%u/jobs/$EXP_NAME/%j.err \
    --partition=learnfair --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:1 \
    --time=00:20:00 \
    --wrap="python ~/lm_code/ngram_lstm.py --test --data $path_lm_data/$lang --save $f --test_path $path_test_data --suffix ngram_lstm_$m --bptt 5 --cuda"
   
    #sleep 5
done

