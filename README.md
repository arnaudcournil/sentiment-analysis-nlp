Reproduction de l'article Learning Word Vectors for Sentiment Analysis

Ce projet vise à reproduire l'approche proposée par Maas et al. (2011) dans leur article Learning Word Vectors for Sentiment Analysis, combinant apprentissage non supervisé probabiliste et supervision pour la polarité des sentiments. Le modèle est évalué sur le jeu de données IMDB.

📌 Objectifs

Implémenter le modèle probabiliste non supervisé proposé pour l'apprentissage de représentations sémantiques.

Intégrer une composante supervisée pour capturer les dimensions de sentiment.

Tester l'effet de la concaténation avec des vecteurs Bag-of-Words (BoW).

Améliorer les performances via un algorithme génétique d'optimisation des hyperparamètres.

🔍 Résultats

Accuracy finale : 88.3% sur la tâche de classification binaire de sentiments (positif/négatif).

Résultats très proches de l'article original (~88.9%).

La combinaison des vecteurs appris avec les BoW fournit les meilleures performances.

⚙️ Architecture du code

notebook.ipynb : entraînement du modèle, tuning par algorithme génétique, évaluation.

article.pdf : article original servant de référence.

🧪 Méthodologie

Pré-traitement du texte :

Suppression des 50 mots les plus fréquents.

Conservation de la ponctuation expressive.

Vocabulaire limité à 5000 tokens.

Pas de stemming.

Entraînement progressif :

Non supervisé seul (peu concluant).

Optimisation d'hyperparamètres via algorithme génétique.

Entraînement conjoint (non supervisé + supervision sentiment).

Ajout des BoW pour obtenir une représentation hybride.

Évaluation :

Classification report avec logistic regression et SVC.

Résultats comparés à ceux de Maas et al.

🧬 Algorithme génétique

Optimise les hyperparamètres .

Évaluation par validation croisée sur un sous-ensemble.

Sélection, croisement, mutation sur plusieurs générations.

📊 Comparaison des configurations

Modèle

Accuracy

Non supervisé seul

~86%

Supervisé seul

~85%

Objectif conjoint

~87.5%

Objectif conjoint + BoW

88.3%

🧭 Branches Git

Voici les étapes effectuées pour réorganiser les branches du dépôt :

# Depuis la branche dev-2nd-model
# 1. Renommer l'ancienne main
git branch -m main old

# 2. Renommer dev-2nd-model en main
git branch -m main

# 3. Pousser la nouvelle main
git push origin -u main

# 4. Supprimer l'ancienne branche main distante
git push origin :main

# 5. Pousser l'ancienne branche locale renommée "old"
git push origin old

📚 Références

Maas et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL. lien

Projet réalisé dans le cadre du master de Data Science.

