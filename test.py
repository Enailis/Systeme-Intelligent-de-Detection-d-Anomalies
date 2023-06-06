import pandas as pd
import os

import pyAgrum.skbn as skbn
import pyAgrum as gum

ot_odr_filename = os.path.join(".", "data", "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join(".", "data", 'EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

var_sig = ["SIG_ORGANE", "SIG_CONTEXTE", "SIG_OBS"]
print(ot_odr_df[var_sig].describe())


# prepare variables

var_cat = ["SIG_OBS", "SYSTEM_N1"]

for i in var_cat:
    ot_odr_df[i] = ot_odr_df[i].astype("category")

ot_odr_df.info()

bn = gum.BayesNet("vroum vroum")
var_to_model = ["SYSTEM_N1", "SIG_OBS"]

var_bn = {}
for var in var_to_model:
    nb_values = len(ot_odr_df[var].cat.categories)
    var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

for var in var_bn:
    for i, modalite in enumerate(ot_odr_df[var].cat.categories):
        var_bn[var].changeLabel(i, modalite)

bn = gum.BayesNet("vroum vroum")

for var in var_bn:
    bn.add(var)

bn.addArc("SIG_OBS", "SYSTEM_N1")

print(bn)

# bn = gum.BayesNet("vroum vroum")

# sig_organe = gum.LabelizedVariable("SIG_ORGANE", "Signalement du conduction sur la partie organe", ot_odr_df['SIG_ORGANE'].unique())
# system_n1= gum.LabelizedVariable("SYSTEM_N1", "Identifiant de système de niveau 1 concerné par l'ODR (niveau macroscopique)", ot_odr_df['SYSTEM_N1'].unique())
# for va in [sig_organe, system_n1]:
#     bn.add(va)

# bn.addArc("SIG_ORGANE", "SYSTEM_N1")

# print(bn)
