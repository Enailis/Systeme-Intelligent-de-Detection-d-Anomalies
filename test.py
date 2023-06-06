from pyagrum_extra import gum

import pandas as pd
import os


###########################################
# Chargement et brève analyse descriptive #
###########################################

ot_odr_filename = os.path.join("./data", "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join("./data", 'EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

# Analyse des modalités des variables de signalement
var_sig = ["SIG_ORGANE", "SIG_CONTEXTE", "SIG_OBS"]
ot_odr_df[var_sig].describe()


# Analyse des modalités des variables systèmes
var_sys = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]
ot_odr_df[var_sys].describe()


# Analyse des modalités des variables type travail et OdR
var_odr = ["TYPE_TRAVAIL", "ODR_LIBELLE"]
ot_odr_df[var_odr].describe()


###########################
# Préparation des données #
###########################

var_cat = ['ODR_LIBELLE', 'TYPE_TRAVAIL',
           'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 
           'SIG_ORGANE', 'SIG_CONTEXTE', 'SIG_OBS', 'LIGNE']
for var in var_cat:
    ot_odr_df[var] = ot_odr_df[var].astype('category')

ot_odr_df.info()

# Création d'un premier modèle
var_to_model = ["SYSTEM_N1", "SYSTEM_N2", "SIG_OBS", "SIG_ORGANE", "SIG_CONTEXTE"]

var_bn = {}
for var in var_to_model:
    nb_values = len(ot_odr_df[var].cat.categories)
    var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

for var in var_bn:
    for i, modalite in enumerate(ot_odr_df[var].cat.categories):
        var_bn[var].changeLabel(i, modalite)

bn = gum.BayesNet("modèle simple")

for var in var_bn.values():
    bn.add(var)

bn.addArc("SYSTEM_N2", "SYSTEM_N1")

bn.addArc("SYSTEM_N1", "SIG_OBS")
bn.addArc("SYSTEM_N1", "SIG_ORGANE")

bn.fit(ot_odr_df)


###############
# Prédictions #
###############

pred_prob = bn.predict_proba(ot_odr_df[["SIG_OBS"]].iloc[-1000:], 
                             var_target="SYSTEM_N1",
                             show_progress=True)
print(pred_prob)

pred = bn.predict(ot_odr_df[["SIG_OBS"]].iloc[-1000:], 
                  var_target="SYSTEM_N1",
                  show_progress=True)

print(pred)