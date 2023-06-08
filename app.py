from pyagrum_extra import gum

import pandas as pd
import os

import dash
from dash import html, dash_table
from dash.dependencies import Input, Output
from dash import dcc
import plotly.express as px

###########################################
# Chargement et brève analyse descriptive #
###########################################

# Chargement des données
ot_odr_filename = os.path.join("./data", "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

# Chargement des données d'équipements
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

# On ne garde que les variables qui nous intéressent
var_cat = ['ODR_LIBELLE', 'TYPE_TRAVAIL',
           'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3',
           'SIG_ORGANE', 'SIG_CONTEXTE', 'SIG_OBS', 'LIGNE']
for var in var_cat:
    ot_odr_df[var] = ot_odr_df[var].astype('category')

ot_odr_df.info()

# Création d'un premier modèle
var_to_model = ["SYSTEM_N1", "SYSTEM_N2", "SIG_OBS", "SIG_ORGANE"]
var_feature = ["SIG_ORGANE", "SIG_OBS"]

var_bn = {}
for var in var_to_model:
    nb_values = len(ot_odr_df[var].cat.categories)
    var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

for var in var_bn:
    for i, modalite in enumerate(ot_odr_df[var].cat.categories):
        var_bn[var].changeLabel(i, modalite)

# Création du réseau bayésien
bn = gum.BayesNet("modèle simple")

# Ajout des noeuds
for var in var_bn.values():
    bn.add(var)

# Ajout des arcs
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

pred = bn.predict(ot_odr_df[["SIG_OBS"]].iloc[-1000:],
                  var_target="SYSTEM_N1",
                  show_progress=True)

print((ot_odr_df["SYSTEM_N1"].iloc[-1000:] == pred).mean())

pred_prob_N2 = bn.predict_proba(ot_odr_df[["SYSTEM_N1"]].iloc[-1000:],
                                var_target="SYSTEM_N2",
                                show_progress=True)

pred_N2 = bn.predict(ot_odr_df[["SYSTEM_N1"]].iloc[-1000:],
                     var_target="SYSTEM_N2",
                     show_progress=True)

print((ot_odr_df["SYSTEM_N2"].iloc[-1000:] == pred_N2).mean())

##################
# Create web app #
##################

top_5 = []
initial_active_cell = {"row": 0, "column": 0, "column_id": "modalité", "row_id": 0}

app = dash.Dash("salut la team")

app.layout = html.Div([
    html.H1("Système Intelligent de Détection d'Anomalies", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.H3(f'{var}'),
            dcc.Dropdown(
                id=f'{var}-dropdown',
                options=[{'label': i, 'value': i} for i in ot_odr_df[var].cat.categories],
                value=ot_odr_df[var].cat.categories[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}) for var in var_feature
    ], style={'width': '100%', 'display': 'inline-block', 'text-align': 'center'}),

    html.Div([
        dash_table.DataTable(
            id='N1_array',
            columns=(
                [{'id': 'modalité', 'name': 'Modalité'},
                 {'id': 'proba', 'name': 'Probabilité'}]
            ),
            data=[],
            editable=True,
            style_cell={'textAlign': 'center'},
            style_cell_conditional=[
                {
                    'if': {'column_id': 'modalité'},
                    'textAlign': 'left'
                }
            ],
            active_cell=initial_active_cell
        ),
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='graph')
    ], style={'width': '75%', 'display': 'inline-block'}
    )
], style={'text-align': 'center'})


@app.callback(
    [Output('N1_array', 'data')],
    [Input(f'{var}-dropdown', 'value') for var in var_feature]
)
def get_N1s(*input):
    global top_5
    bn_ie = gum.LazyPropagation(bn)

    ev = {var: value for var, value in zip(var_feature, input)}
    bn_ie.setEvidence(ev)
    bn_ie.makeInference()

    var_targets = ["SYSTEM_N1"]

    temp_array = None
    for var in var_targets:
        temp_array = bn_ie.posterior(var).topandas().droplevel(0).sort_values(ascending=False).head(5)

    top_5 = temp_array.index.tolist()
    temp_array = temp_array.reset_index()

    temp_array.columns = ["modalité", "proba"]

    return [temp_array.to_dict('records')]


@app.callback(
    [Output('graph', 'figure')],
    [Input('N1_array', 'active_cell')]
)
def update_graphs(active_cell):
    global top_5
    if active_cell != None:
        bn_ie = gum.LazyPropagation(bn)

        ev = {}
        ev["SYSTEM_N1"] = top_5[active_cell['row']]
        bn_ie.setEvidence(ev)
        bn_ie.makeInference()

        prob_target = []
        prob_target_var = bn_ie.posterior("SYSTEM_N2").topandas().droplevel(0).sort_values(ascending=False)
        prob_target_var = prob_target_var[prob_target_var != 0]
        prob_fig = px.bar(prob_target_var)
        prob_target.append(prob_fig)

        return tuple(prob_target)


app.run_server(debug=True, port=8086)