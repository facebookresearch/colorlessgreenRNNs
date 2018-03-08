## Data for model training and evaluation

- `agreement` contains evaluation data generated based on long distance agreement patterns
- `extended_test_data` contains evaluation data and results of our trained models
- `linzen_testset` contains the subset of data from *Linzen et al. TACL 2016* (https://github.com/TalLinzen/rnn_agreement) which we used for our evaluation
- `raw_mturk_data` contains the reponses of MTurk subjects for the extended Italian agreement data 

### Training data based on Wikipedia

Each corpus consists of around 100M tokens, we used training (80M) and validation (10M) subsets in our experiments. All corpora were shuffled at sentence level.

- English [train](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/train.txt) / [valid](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/valid.txt) / [test](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/test.txt) / [vocab](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/vocab.txt)
- Hebrew [train](https://s3.amazonaws.com/colorless-green-rnns/training-data/Hebrew/train.txt) / [valid](https://s3.amazonaws.com/colorless-green-rnns/training-data/Hebrew/valid.txt) / [test](https://s3.amazonaws.com/colorless-green-rnns/training-data/Hebrew/test.txt) / [vocab](https://s3.amazonaws.com/colorless-green-rnns/training-data/Hebrew/vocab.txt)
- Italian [train](https://s3.amazonaws.com/colorless-green-rnns/training-data/Italian/train.txt) / [valid](https://s3.amazonaws.com/colorless-green-rnns/training-data/Italian/valid.txt) / [test](https://s3.amazonaws.com/colorless-green-rnns/training-data/Italian/test.txt) / [vocab](https://s3.amazonaws.com/colorless-green-rnns/training-data/Italian/vocab.txt)
- Russian [train](https://s3.amazonaws.com/colorless-green-rnns/training-data/Russian/train.txt) / [valid](https://s3.amazonaws.com/colorless-green-rnns/training-data/Russian/valid.txt) / [test](https://s3.amazonaws.com/colorless-green-rnns/training-data/Russian/test.txt) / [vocab](https://s3.amazonaws.com/colorless-green-rnns/training-data/Russian/vocab.txt)

### Pre-trained language models

For each language, we distribute the trained LSTM model which achieved the lowest perplexity on our test set (validation in the data above). The name of the model file indicates the hyperparameters that were used to train this model. See the supplementary materials for more details.

The models were trained with the vocabularies given above. Each vocabulary lists words according to their indices starting from 0, `<unk>` and `<eos>` tokens are already in the vocabulary.

* [English model](https://s3.amazonaws.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt)
* [Hebrew model](https://s3.amazonaws.com/colorless-green-rnns/best-models/Hebrew/hidden650_batch64_dropout0.1_lr20.0.pt)
* [Italian model](https://s3.amazonaws.com/colorless-green-rnns/best-models/Italian/hidden650_batch64_dropout0.2_lr20.0.pt)
* [Russian model](https://s3.amazonaws.com/colorless-green-rnns/best-models/Russian/hidden650_batch64_dropout0.2_lr20.0.pt)

-------

Please cite the paper if you use the above resources.