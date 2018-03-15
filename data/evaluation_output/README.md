## Evaluation data and results

This directory contains the long-distance agreement data, together with model and human subject performance.

Each language directory contains two main files  `best_models.tab` and `all_models.tab`.

For Italian, we provide the file `best_model_mturk.tab` containing the Italian data, together with the results for the best LSTM model and aggregated subject performance. The file contains the following tab-delimited fields 

1. `pattern`  construction (defined by the sequence of PoS tags) the current sentence instantiates

2. `constr_id`  id of the reference original sentence

4. `sent_id`  id for the specific sentence, 0 corresponds to the original reference sentence, 1-10 to nonce generated sentence 

4) `correct_number`  whether the correct target should be sing or plur

5) `form`  candidate target

6) `class`  whether the target is correct or wrong

7) `type`  whether the sentence is original or generated (that is, nonce)

8) `prefix`  the sentence up to and excluding the target

9) `n_attr`  number of attractors

10) `punct`  does context include punctuation

11) `freq`  frequency of target form in training corpus

12) `len_context`  number of words in context

13) `len_prefix`  number of words from beginning of sentence up to and excluding target

14) `sent`  this field contains information used in the process to construct stimuli, and should be ignored

15) `full_id`  unique id for the sentence (shared across candidate targets)

16) `mturk_count`  number of subjects who preferred the current target for this sentence

17) `hidden650_batch64_dropout0.2_lr10.0`  probability assigned to this target by best LSTM

The `best_models.tab` files report the same data as above for all the languages, but do not not incude subject data (the header of the best LSTM model columns corresponds to the hyperparameter choices for the relevant language).

The `all_models.tab` contains the same data, with columns for all the LSTM models.

----

The corresponding data for the n-gram LSTMs and simple RNNs baselines are available as `best_model_ngram_lstm.tab`, `all_models_ngram_lstm.tab`, `best_model_srnn.tab` and `all_models_srnn.tab` respectively.
