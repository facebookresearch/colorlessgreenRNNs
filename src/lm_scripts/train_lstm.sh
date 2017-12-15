#module load pytorch/0.1.8-py27_2-anaconda2.4.3.1 

lang=$2

#dir=lstm_hyperparameters_exps/Hebrew/

dir=lstm_hyperparameters_exps/$lang

#dir=lstm_hyperparameters_exps/English_seeds/
#dir=lstm_hyperparameters_exps/Italian_srnn/
#data_dir=/private/home/gulordava/edouard_data/itwiki/50K_vocab_shuffled/
#data_dir=/private/home/gulordava/edouard_data/enwiki/50K_vocab/

repo=/private/home/gulordava/colorlessgreen/data/lm/
data_dir=$repo$lang

while read name nhidden batch_size dropout lr
do 
  if [[ $name == "#"* ]] ; then
    continue
  fi
  name=${name}
  EXP_NAME=$dir/$name
  EXP_DIR=/private/home/gulordava/$dir

  mkdir -p /checkpoint/$USER/jobs/$EXP_NAME
  mkdir -p $EXP_DIR/models
  mkdir -p $EXP_DIR/logs

  echo $name $nhidden $batch_size $dropout $lr

  sbatch --job-name=$EXP_NAME \
    --output=/checkpoint/%u/jobs/$EXP_NAME/%j.out \
    --error=/checkpoint/%u/jobs/$EXP_NAME/%j.err \
    --partition=learnfair --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:1 \
    --time=48:00:00 \
    --wrap="export PATH=$PATH:/public/slurm/17.02.6/bin; srun /private/home/gulordava/anaconda3/bin/python -u /private/home/gulordava/lm_code/main.py --cuda --data $data_dir --save $EXP_DIR/models/${name}.pt --log $EXP_DIR/logs/${name}.log --batch_size $batch_size --lr $lr --nhid $nhidden --emsize $nhidden --dropout $dropout --epochs 40"
done < $1

