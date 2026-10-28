

\# Optimisation de la fertilisation agricole par un système multi-agents



\## 1. Présentation du projet



Ce projet propose un prototype de système multi-agents (SMA) pour l'optimisation de la fertilisation agricole à l'aide de l'apprentissage par renforcement.



L'objectif est d'étudier comment des agents autonomes peuvent apprendre à choisir des doses de fertilisation adaptées aux caractéristiques des sols.



\## 2. Problématique



Une fertilisation inadaptée aux caractéristiques du sol peut entraîner une utilisation inefficace des engrais et un déséquilibre nutritionnel.



Le projet explore une approche fondée sur les données pédologiques et l'apprentissage par renforcement afin d'améliorer la prise de décision en matière de fertilisation.



\## 3. Approche proposée



Le prototype utilise notamment :



\- \*\*Système multi-agents (SMA)\*\* : organisation de la prise de décision autour d'agents autonomes.

\- \*\*Double Q-learning\*\* : méthode d'apprentissage par renforcement utilisée pour apprendre une politique de décision.

\- \*\*Données pédologiques\*\* : cartes de phosphore disponible et de potassium échangeable, ainsi que d'autres variables environnementales.

\- \*\*Visualisation\*\* : interface HTML et JavaScript permettant de présenter des éléments du projet.



\## 4. Technologies utilisées



\- Python

\- NumPy

\- Matplotlib

\- Scikit-learn

\- HTML

\- JavaScript

\- Double Q-learning



\## 5. Structure du projet



\- `fertilisation\_sma.py` : code principal du modèle.

\- `robustesse.py` : expériences et analyses de robustesse.

\- `application.html` : interface de visualisation.

\- `donnees.js` : données utilisées par l'interface.

\- `resultats.js` : résultats destinés à l'affichage.

\- `robustesse.js` : éléments JavaScript associés à la robustesse.

\- `requirements.txt` : dépendances Python.

\- `\*.tif` : fichiers de données géospatiales.



\## 6. Installation



Python doit être installé sur la machine.



Cloner le dépôt :



```bash

git clone URL\_DU\_DEPOT

cd NOM\_DU\_DEPOT

```



Créer un environnement virtuel :



```bash

python -m venv .venv

```



Activer l'environnement sous Windows :



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Installer les dépendances :



```bash

python -m pip install -r requirements.txt

```



\## 7. Exécution



Lancer le programme principal :



```bash

python fertilisation\_sma.py

```



Exécuter les expériences de robustesse :



```bash

python robustesse.py

```



L'interface peut être ouverte dans un navigateur à partir du fichier `application.html`.



Les commandes et dépendances devront être confirmées selon la configuration finale du projet.



\## 8. Données



Le projet contient des fichiers géospatiaux au format GeoTIFF (`.tif`).



Avant leur redistribution, il faut vérifier leurs sources, leurs licences et leurs conditions d'utilisation.



Les fichiers volumineux peuvent être distribués séparément, avec des instructions permettant de les obtenir.



\## 9. Limites



Ce dépôt présente un prototype expérimental. Les performances de l'apprentissage, la robustesse des décisions et la pertinence agronomique des doses doivent être évaluées à partir de résultats reproductibles.



Les résultats simulés ne constituent pas, à eux seuls, une validation agronomique sur le terrain.



\## 10. Perspectives



\- Améliorer l'évaluation des politiques apprises.

\- Comparer les décisions apprises à des stratégies de référence.

\- Étudier l'influence des paramètres d'apprentissage.

\- Renforcer la validation sur des données agricoles appropriées.

\- Améliorer la visualisation et la documentation du système.



\---



Projet académique — Processus stochastiques et systèmes multi-agents.



