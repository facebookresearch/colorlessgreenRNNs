This directory contains the scripts we used to run our experiments:

- the `syntactic_testsets` directory contains scripts to extract constructions and generate nonce sentences
  - please check  `create_testset.sh` to see how these scripts are used and put together
- the `language_models` directory contains scripts for training and evaluating language models (standard and n-gram LSTMs) using PyTorch, based on the code [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)
- the `data` directory contains scripts for treebank preprocessing



. TODO



- `collect_results.py` takes the outputs of the models and produces the files in the `data/results` directory
- `evaluate.py` takes files in the `data/results` files and generates summary accuracies
- `evaluate_by_pattern.py` outputs by-constrution accuracies