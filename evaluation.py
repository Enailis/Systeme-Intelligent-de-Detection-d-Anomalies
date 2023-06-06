import pandas as pd
import os

########################
# Evaluation du modèle #
########################

ot_odr_filename = os.path.join(".", "data", "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

# On génère deux ensembles
ot_odr_df_train = ot_odr_df.iloc[:-10000] # Toutes les premières données dans train
ot_odr_df_test  = ot_odr_df.iloc[-10000:] # Les 10 000 dernières dans test

# On fit avec les données de train
bn.fit(ot_odr_df_train)

# On prédit la variable cible
pred = bn.predict(ot_odr_df_test[["SIG_OBS"]], var_target="SYSTEM_N1", show_progress=True)

# Affichage du taux de bonne prédictions
print((ot_odr_df_test["SYSTEM_N1"] == pred).mean())