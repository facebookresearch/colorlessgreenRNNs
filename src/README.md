This directory contains the scripts we used to run our experiments:

- the `syntactic_testsets` directory contains scripts to extract constructions and generate nonce sentences
  - please check  `create_testset.sh` to see how these scripts are used and put together
- the `language_models` directory contains scripts for training and evaluating language models (standard and n-gram LSTMs) using PyTorch (0.3), based on the code [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)
- the `data` directory contains scripts for treebank preprocessing



### Usage example

*works with python 3.5, pandas 0.22*

#### Creating evaluation datasets

Download LM Wikipedia data to `../data/lm/English/` and run `create_testset.sh` with default paths to obtain evaluation data files with prefix `tmp/English/generated`. 

Our generated data tested in the paper is in `../data/agreement/English/generated`. (Since random substitution for nonce sentences is used, these files won't be identical to newly produced files, but the original sentences extracted should be the same if you use the English UD 2.0 treebank.)

#### Evaluating a language model

You can use your trained `model.pt` or download our [pre-trained best model](../data).

Run the command (`--cuda` is optional)

```bash
python language_models/evaluate_target_word.py --data ../data/lm/English/ --checkpoint model.pt --path ../data/agreement/English/generated --suffix best_model --cuda
```

to produce the tab-delimited file `../data/agreement/English/generated.output_best_model` with probabilities for target word positions. The value of `--suffix` argument is used as the name of the column containing model probabilities.

Run `python results.py English best_model` to obtain a summary of performance for `best_model` (the column name) extracted from the output file (works with default paths). 

*Note:* this workflow made sense for our evaluations but is not the most direct way to obtain accuracy numbers. We might provide a simplified script for evaluation in the future.





