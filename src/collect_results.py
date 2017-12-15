
import os
import sys

import pandas as pd

from syntactic_testsets.utils import load_vocab


def get_best_val_score(f_name):
    ppls = []
    with open(f_name, "r") as f:
        for line in f:
            if "valid" in line:
                fields = [f for f in line.strip().split("|") if f != ""][1:]
                values = [f.split()[-1] for f in fields]
                ppls.append(float(values[-1]))
    if len(ppls) == 0:
        return 0
    return min(ppls)


def get_oov_valid_score(f_name):
    """ valid score without OOV """
    with open(f_name, "r") as f:
        for line in f:
            if "test " in line:
                fields = [f for f in line.strip().split("|") if f != ""][1:]
                values = [f.split()[-1] for f in fields]
                return float(values[-1])


def lstm_probs(output, gold, w2idx):
    data = []
    for scores, g in zip(output, gold):
        scores = scores.split()
        form, form_alt = g.split("\t")[6:8]

        prob_correct = float(scores[w2idx[form]])
        prob_wrong = float(scores[w2idx[form_alt]])

        data.append(prob_correct)
        data.append(prob_wrong)

    return data


lang = sys.argv[1]

path = "/private/home/gulordava/lstm_hyperparameters_exps/" + lang

model_type = ""
if len(sys.argv) > 2:
    model_type = sys.argv[2]

path = path + "/" + model_type

path_repo = "/private/home/gulordava/colorlessgreen/data"
path_test_data = path_repo + "/agreement/" + lang + "/generated"
path_output = path_repo + "/agreement/" + lang + "/generated.output_"
path_lm_data = path_repo + "/lm/" + lang


path_models = path + "/models/"
if model_type == "ngram_lstm":
    path_logs = path + "/logs/"
else:
    path_logs = path + "/valid_ppls/"


log_files = os.listdir(path_logs)

models_ppls = {}
for f in log_files:
    val_ppl = get_oov_valid_score(path_logs + f)
    if val_ppl:
        if model_type:
            m = "_".join([model_type, f[:-4]])
        else:
            m = f[:-4]
        models_ppls[m] = val_ppl
# best models
print("Best models:")
for m in sorted(models_ppls, key=models_ppls.get)[:10]:
    print(m, models_ppls[m])

best_model = sorted(models_ppls, key=models_ppls.get)[0]
print("Best model", best_model)

models = list(models_ppls.keys())

gold = open(path_test_data + ".gold").readlines()
sents = open(path_test_data + ".text").readlines()
data = pd.read_csv(path_test_data + ".tab",sep="\t")

vocab = load_vocab(path_lm_data + "/vocab.txt")

# getting softmax outputs and the probabilities for pairs of test forms
print("Reading softmax output for", len(models), "models from", path_output)
print("Assembling probabilities for the choice forms")
outputs = {}
probs = pd.DataFrame([])
for m in models:
    # normally, don't need .pt
    if os.path.isfile(path_output + m):
        print(m)
        output = open(path_output  + m).readlines()
        print(len(output))
        data[m] = lstm_probs(output, gold, vocab)

models = list(outputs.keys())

print("All models", len(data))

#input_data = path_repo + "agreement/" + lang + ".tab"
#print("Saving input data (wo models) to", input_data)

fields = [f for f in data.columns if "hidden" not in f]

data.to_csv(path_repo + "/results/" + model_type + "/" + lang + "/all_models.tab",sep="\t",index=False)
data[fields + [best_model]].to_csv(path_repo + "/results/" + model_type + "/" + lang + "/best_model.tab",sep="\t",index=False)