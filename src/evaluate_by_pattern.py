import pandas as pd
import sys

pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('max_colwidth',150)
pd.set_option('display.width', 200)

lang = sys.argv[1]
if len(sys.argv) > 2:
    model_type = sys.argv[2]
else:
    model_type = "" 

# getting results for the best model
full_df = pd.read_csv("/private/home/gulordava/colorlessgreen/data/results/" + model_type + "/" + lang + "/best_model.tab",sep="\t")

fields = ["pattern","constr_id","sent_id","n_attr","punct","len_prefix","len_context","sent","correct_number","type"]

# our "models" - best model and unigram baseline for comparison
models = [f for f in full_df.columns if "hidden" in f] + ["freq"]

wide_data = full_df[fields + ["class"] + models].pivot_table(columns=("class"),values=(models),index=fields)

for model in models:
    correct = wide_data.loc[:, (model, "correct")]
    wrong = wide_data.loc[:, (model, "wrong")]
    wide_data[(model, "acc")] = (correct > wrong)*100

t = wide_data.reset_index()
a = pd.concat([t[t.type=="original"].groupby("pattern").agg({(m, "acc"): "mean" for m in models}).rename(columns={'acc': 'orig'}),
               t[t.type=="generated"].groupby("pattern").agg({(m, "acc"): "mean" for m in models}).rename(columns={'acc': 'gen'})],
              axis=1)
#a.index = ms
#a = a.reset_index()
#a.columns = ["pattern"] + ["acc_original_" + m for m in models] + ["acc_generated_" + m for m in models]


print(a)


