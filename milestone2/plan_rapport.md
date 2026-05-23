# Plan du Rapport — Milestone 2

Structure calquée sur le M1 (2 pages, double colonne, anglais).

---

## 1. Introduction
- Même dataset que M1 (2000 samples, 13 features, 1600 train / 400 test)
- Nouvelles méthodes : MLP (classification + régression) + K-Means (classification)
- Deux tâches : prédire addiction level (classification) et addiction score (régression)

---

## 2. Methodology

### 2.1 Data Preparation
- Z-score normalisation (µ et σ calculés sur train uniquement, pas de data leakage)
- Split 80/20 avec **permutation aléatoire** avant découpage ← mentionner explicitement (demandé dans feedback M1)

### 2.2 MLP
- Architecture : `(13, 64, C)` avec 1 couche cachée
- Activation cachée : ReLU — Activation sortie : Sigmoid (classification) / Identity (régression)
- Loss : CrossEntropy (classification) / MSE (régression)
- Initialisation Xavier / Glorot
- Expliquer le choix Sigmoid plutôt que Softmax en sortie (gradient element-wise compatible avec le backprop implémenté)
- Mini-batch SGD avec shuffle à chaque epoch

### 2.3 K-Means
- Algorithme standard, initialisation aléatoire, critère de convergence `np.allclose`
- Extension à la classification : assignation d'un label à chaque cluster par vote majoritaire
- Prédiction : point → cluster le plus proche → label du cluster

---

## 3. Experiments & Results

### 3.1 Hyperparameter Tuning

**MLP — Heatmap 2D lr × epochs** ← demandé explicitement dans le feedback M1
- Axes : lr ∈ {5e-4, 1e-3, 3e-3} × epochs ∈ {100, 200, 300, 500}
- Métrique affichée : Accuracy (classification) et MSE (régression)
- Permet de voir conjointement l'effet des deux hyperparamètres

**K-Means — Courbe F1 vs K**
- K testé : {3, 5, 9, 12, 15, 20, 25, 30}, moyenné sur 10 seeds
- Même style que le graphe KNN du M1
- Montrer que K=3 (= nb de classes) donne le même résultat que le Dummy Classifier

### 3.2 Final Test Set Evaluation
Tableau unique (style M1) — tous les modèles retrained sur les 1600 samples complets :

| Method | HP | Acc. | F1 | MSE |
|---|---|---|---|---|
| Dummy Classifier | — | 69.69% | 0.274 | — |
| K-Means | K=20 | ~73.7% | ~0.56 | — |
| MLP | lr=3e-3, ep=300, h=64 | ~80.5% | ~0.48 | — |
| Dummy Classifier | — | — | — | ~? |
| MLP | lr=1e-3, ep=200, h=64 | — | — | ~1.03 |

---

## 4. Discussion & Conclusion

- **KMeans vs MLP** : KMeans K=20 surpasse le MLP en Macro F1 (0.56 vs 0.48) malgré une accuracy plus basse (73.7% vs 80.5%) — le vote majoritaire avec beaucoup de clusters détecte mieux les classes minoritaires
- **Overfitting MLP régression** : MSE augmente au-delà de 200 epochs → le modèle overfite rapidement sur ce petit dataset
- **Comparaison avec M1** : Logistic Regression atteignait 88.25% / F1 0.819 → le MLP (80.5% / F1 0.48) est moins bon → argument : les réseaux de neurones ne gagnent pas toujours contre des méthodes plus simples sur de petits datasets
- Conclusion : K-Means reste compétitif pour la classification grâce au vote majoritaire ; le MLP nécessite un tuning soigné et reste sensible à l'overfitting en régression

---

## Notes pratiques

- **Langue** : tout en anglais — code (commentaires, prints) + rapport (feedback M1)
- **Figures à générer** :
  1. Heatmap lr × epochs pour MLP (classification)
  2. Courbe F1 vs K pour KMeans
- **Bonus cross-validation** (+1pt possible) : coûteux avec MLP (300 epochs × 5 folds) — à faire si le temps le permet
- **Soumission** : ne pas inclure le dossier `venv/` ni le fichier `data/features.npz` (feedback M1 : 122 MB)
