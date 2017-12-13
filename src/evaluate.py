

import pandas as pd
pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('max_colwidth',150)
pd.set_option('display.width', 200)

import sys
import os

def get_best_test_score(f_name):
    ppls = []
    with open(f_name, "r") as f:
        for line in f:
            if "test " in line:
                fields = [f for f in line.strip().split("|") if f != ""][1:]
                values = [f.split()[-1] for f in fields]
                return float(values[-1])            


lang = sys.argv[1]
if len(sys.argv) > 2:
    model_type = sys.argv[2]
else:
    model_type = ""

path_repo = "/private/home/gulordava/colorlessgreen/data"
path_test_data = path_repo + "/agreement/" + lang + "/generated"
path_lm_data = path_repo + "/lm/"

path = "/private/home/gulordava/lstm_hyperparameters_exps/" + lang + "/" + model_type

path_models = path + "/models/"
# indicate the files with the perplexities of the models
if model_type == "ngram_lstm":
    path_logs = path + "/logs/"
# wo OOV words
else:
    path_logs = path + "/valid_ppls/"

log_files = os.listdir(path_logs)
print(path_logs)

models_ppls = {}
for f in log_files:
    val_score = get_best_test_score(path_logs + f)
    if val_score:
        if model_type:
            m = "_".join([model_type, f[:-4]])
        else:
            m = f[:-4]
        models_ppls[m] = val_score
# best models
for f in sorted(models_ppls, key=models_ppls.get)[:10]:
    print(f, models_ppls[f])

path_results = "/private/home/gulordava/colorlessgreen/data/results/" + model_type + "/" + lang + "/" 
full_df = pd.read_csv(path_results + "all_models.tab",sep="\t")

fields = ["pattern","constr_id","sent_id","n_attr","punct","len_prefix","len_context","sent","correct_number","type"]
models = [f for f in full_df.columns if "hidden" in f]  + ["freq"]

wide_data = full_df[fields + ["class"] + models].pivot_table(columns=("class"),values=models,index=fields)

for model in models:
    correct = wide_data.loc[:, (model, "correct")]
    wrong = wide_data.loc[:, (model, "wrong")]
    wide_data[(model, "acc")] = correct > wrong

t = wide_data.reset_index()
a = pd.concat([t.agg({(m,"acc"):"mean" for m in models}), 
           t[t.sent_id==0].agg({(m,"acc"):"mean" for m in models}),
           t[t.sent_id!=0].agg({(m,"acc"):"mean" for m in models})],
          axis=1)
a.index = models
a = a.reset_index()
a.columns = ["model","acc","acc_original","acc_generated"]
a.acc = a.acc * 100
a.acc_original = a.acc_original * 100
a.acc_generated = a.acc_generated * 100

# adding perplexity information
a["ppls"] = a["model"].map(models_ppls)

print("Top 10 models by perplexity")
print(a.sort_values("ppls")[:10].to_string(index=False))
a.sort_values("ppls")[:10].to_csv(path_results + "accuracy_top10_ppls.tab",sep="\t")

print("Most frequent baseline")
print(a[a.model == "freq"].to_string(index=False))
