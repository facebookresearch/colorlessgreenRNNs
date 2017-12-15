#input file_all_models path_lm_data path_models path_test_data

lang=$1
model_type="srnn"
models_path="/private/home/gulordava/lstm_hyperparameters_exps/$lang/$model_type/models"
path_lm_data="/private/home/gulordava/colorlessgreen/data/lm/$lang"

path_test_data="/private/home/gulordava/colorlessgreen/data/agreement/$lang/generated"

for f in $models_path/* ; do
    m=`basename $f`
    m="${m%.*}"
    m=${model_type}_$m
    echo $m

    EXP_NAME=eval_$lang/$m

    mkdir -p /checkpoint/$USER/jobs/$EXP_NAME/
    

    sbatch --job-name=$EXP_NAME \
    --output=/checkpoint/%u/jobs/$EXP_NAME/%j.out \
    --error=/checkpoint/%u/jobs/$EXP_NAME/%j.err \
    --partition=learnfair --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:1 \
    --time=00:30:00 \
    --wrap="python ~/lm_code/evaluate_target_word.py --data $path_lm_data --checkpoint $f --path $path_test_data --suffix $m --cuda > /dev/null"
    #python ~/lm_code/evaluate_target_word.py --data $path_lm_data --checkpoint $f --path $path_test_data --suffix $m --cuda
done

