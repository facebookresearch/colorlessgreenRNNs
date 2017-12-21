This directory contains the script we used to run our experiments.

- The `data` directory contains scripts for Treebank preprocessing.

- The `language_models` directory contains scripts for training and evaluating language models (standard and n-gram LSTMs) based on [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model).

- The `lm_scripts` are scripts for grid search and batch processes in general.

- The `syntactic_testsets` directory contains scripts to extract constructions and generate nonce sentences. The `create_testset.sh` scripts assembles these steps.

- `collect_results.py` takes the outputs of the models and produces the files in the `data/results` directory.

- `evaluate.py` takes files in the `data/results` files and generates summary accuracies.

- `evaluate_by_pattern.py` outputs by-constrution accuracies.

- `train_ngram_lm.sh` should be moved to `lm_scripts`.
