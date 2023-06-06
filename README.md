# Systeme Intelligent de Detection d'Anomalies (SIDA)

Le Systeme Intelligent de Detection d'Anomalies (SIDA) vous permet d'améliorer l'efficacité des opérations de maintenance sur les véhicules en élaborant des recommandations pour les mainteneurs.

Le SIDA est un outil d'aide à la décision pour déterminer les Ordres De Réparation (ODR) les plus appropriés par rapport aux données de signalement.

## Installation

Voici les procédures d'installation du SIDA sur votre machine.

1. Cloner le dépôt GitHub
2. Créer un dossier `data`
3. Y placer les fichiers `EQUIPEMENTS.csv` et `OR_ODR.csv.bz2` contenant vos données d'historique de maintenance
4. Installer les dépendances Python : `pip install -r requirements.txt`

## Utilisation

Pour démarrer le SIDA, il faut lancer le fichier `app.py`.

## Données d'entrée :

Notre application utilise en entrée un fichier csv qui regroupe un ensemble de données sur l'historique des anomalies détectées sur les véhicules de l'entreprise.
Dans celui-ci, on y associe les remarques des techniciens de maintenance sur les anomalies détectées et les actions de maintenance effectuées.
On y retrouve aussi d'autre information comme le kilométrage du véhicule et la ligne suivi par celui-ci.

## Fonctionnement

On se base sur une approche probabiliste en utilisant les réseaux bayésiens.

## Modélisation

## Evaluation des performances
