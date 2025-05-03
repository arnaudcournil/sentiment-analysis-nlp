# Sentiment-Aware Word Embeddings — Reproduction Study

Ce dépôt contient la reproduction complète du modèle présenté dans l’article **"Learning Word Vectors for Sentiment Analysis"** (Maas et al., ACL 2011), enrichie de nos propres expérimentations et améliorations.

## 🔍 Objectif

Reproduire le modèle hybride de Maas et al. combinant :
- **Apprentissage non supervisé** : un modèle log-linéaire probabiliste apprenant des représentations de mots à partir de documents.
- **Apprentissage supervisé** : une régression logistique apprenant à prédire la polarité d’un document à partir des vecteurs de mots.

## 📑 Approche suivie

1. **Prétraitement du corpus IMDB**
   - Suppression des 50 mots les plus fréquents
   - Conservation des ponctuations expressives (`!`, `:-)`), importantes pour la polarité
   - Vocabulaire limité à 5000 tokens
   - Aucun stemming appliqué

2. **Premiers essais**
   - Entraînement séparé des composantes : non supervisé seul, puis supervision sur les vecteurs appris
