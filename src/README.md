This directory contains the scripts we used to run our experiments:

- the `syntactic_testsets` directory contains scripts to extract constructions and generate nonce sentences
  - please check  `create_testset.sh` to see how these scripts are used and put together
- the `language_models` directory contains scripts for training and evaluating language models (standard and n-gram LSTMs) using PyTorch (0.3), based on the code [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)
- the `data` directory contains scripts for treebank preprocessing



### Usage example

*works with python 3.5, pandas 0.22*

Download LM Wikipedia data to `../data/lm/English/` and run `create_testset.sh` with default paths to obtain evaluation data files with prefix `tmp/English/generated`. 

Our generated data tested in the paper is in `../data/agreement/English/generated`. (Since random substitution for nonce sentences is used, these files won't be identical to newly produced files, but the original sentences extracted should be the same if you use the English UD 2.0 treebank.)

You can use your trained `model_x.pt` or download our pre-trained best model.

Run the command

```bash
python language_models/evaluate_target_word.py --data ../data/lm/English/ --checkpoint model_x.pt --path ../data/agreement/English/generated --suffix model_x --cuda
```

to produce the file `../data/agreement/English/generated.output_model_x` with probabilities for target word positions.

Run `python results.py English model_x` to obtain a summary of performance of the `model_x` extracted from the output file (works with default paths). 

*Note:* this workflow made sense for our evaluations but is not the most direct way to obtain accuracy numbers. We might provide a simplified script for evaluation in the future.





