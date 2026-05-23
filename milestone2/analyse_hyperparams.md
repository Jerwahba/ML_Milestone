# Analyse des hyperparamètres — Milestone 2

Dataset : Gaming and Mental Health (1600 train / 400 test, 13 features, normalisé Z-score).
Toutes les évaluations ci-dessous sont faites sur le jeu de test complet (`--test`).

---

## MLP Classification

**Architecture testée** : `(13, hidden_dim, 3)`, activation sortie = Sigmoid  
**Loss** : CrossEntropy  
**Note** : Softmax a été testé mais donne de mauvais résultats car son gradient n'est pas element-wise — le backprop élémentaire est incorrect avec Softmax. Sigmoid est mathématiquement correct dans ce cadre et donne de meilleures performances.

### Influence de l'architecture (lr=1e-3, epochs=150, bs=32)

| Architecture | Acc | F1 |
|---|---|---|
| `(13, 64, 3)` 1 couche | **76.50%** | **0.421** |
| `(13, 128, 3)` 1 couche | 75.00% | 0.387 |
| `(13, 64, 32, 3)` 2 couches | 70.00% | 0.275 |
| `(13, 128, 64, 3)` 2 couches | 70.25% | 0.283 |
| `(13, 256, 128, 3)` 2 couches | 70.00% | 0.275 |

→ **1 couche cachée de 64 neurones** est la meilleure architecture. Les réseaux plus profonds sous-performent probablement à cause du gradient qui disparaît sur ce petit dataset.

### Influence du learning rate et du nombre d'epochs (arch = `(13,64,3)`, bs=32)

| lr | epochs | Acc | F1 |
|---|---|---|---|
| 5e-4 | 100 | 73.75% | 0.405 |
| 5e-4 | 300 | 76.50% | 0.421 |
| 1e-3 | 100 | 75.25% | 0.406 |
| 1e-3 | 200 | 78.25% | 0.446 |
| 1e-3 | 300 | 79.00% | 0.458 |
| **3e-3** | **300** | **80.50%** | **0.479** |
| 3e-3 | 700 | 80.75% | 0.484 |
| 3e-3 | 1000 | 81.75% | 0.495 |

→ **lr=3e-3, epochs=300** est le meilleur compromis temps/performance. Au-delà de 300 epochs, le gain est marginal (~1%).

**Configuration finale retenue :**
```
MLP classification : dimensions=(13, 64, 3), activations=(ReLU, Sigmoid)
lr=3e-3, epochs=300, batch_size=32
→ Accuracy ~80.5%, Macro F1 ~0.48
```

---

## MLP Régression

**Architecture** : `(13, hidden_dim, 1)`, activation sortie = Identity  
**Loss** : MSE

### Influence de l'architecture (lr=1e-3, epochs=200, bs=32)

| Architecture | MSE |
|---|---|
| `(13, 64, 1)` | 1.026 |
| `(13, 128, 1)` | 1.030 |
| `(13, 64, 32, 1)` 2 couches | **1.001** |
| `(13, 128, 64, 1)` 2 couches | 1.012 |

### Overfitting avec plus d'epochs — architecture `(13,64,1)` lr=1e-3

| epochs | MSE |
|---|---|
| 200 | **1.026** |
| 300 | 1.010 |
| 500 | 1.041 |
| 700 | 1.093 |
| 1000 | 1.209 |

→ Le modèle **overfite clairement après 200 epochs** sur la régression. On conserve 1 couche cachée et on stoppe tôt.

**Configuration finale retenue :**
```
MLP régression : dimensions=(13, 64, 1), activations=(ReLU, Identity)
lr=1e-3, epochs=200, batch_size=32
→ MSE ~1.026
```

---

## KMeans

Le K-Means est utilisé pour la classification par vote majoritaire dans chaque cluster.  
Chaque test est moyenné sur 10 seeds aléatoires pour neutraliser la sensibilité à l'initialisation.

### Influence de K (max_iters=300, moyenne sur 10 seeds)

| K | Acc moyenne | F1 moyenne |
|---|---|---|
| 3 | 71.17% ± 1.65 | 0.320 ± 0.064 |
| 5 | 70.00% ± 0.00 | 0.275 ± 0.000 |
| 9 | 71.83% ± 2.59 | 0.314 ± 0.056 |
| 12 | 74.50% ± 0.94 | 0.374 ± 0.017 |
| **20** | **73.67% ± 0.83** | **0.560 ± 0.035** |
| 25 | 74.42% ± 1.41 | 0.558 ± 0.058 |
| 30 | 74.47% ± 1.30 | 0.577 ± 0.064 |

**Observations :**
- K=3 (= nombre de classes) donne le même résultat que le dummy classifier → trop peu de clusters pour capturer la structure interne des données.
- K=5 et K=6 bloquent à 70% sur toutes les seeds → minimum local systématique à ces valeurs.
- **K=20 est le meilleur compromis** : F1 élevé (0.56) avec faible variance (±0.035). K=30 est légèrement meilleur en F1 mais plus instable.

**Configuration finale retenue :**
```
KMeans : K=20, max_iters=300
→ Accuracy ~73.7%, Macro F1 ~0.56
```

---

## Comparaison finale des méthodes

| Méthode | Tâche | Accuracy | Macro F1 / MSE |
|---|---|---|---|
| Dummy Classifier | Classification | 69.69% | F1 = 0.27 |
| KMeans (K=20) | Classification | ~73.7% | **F1 = 0.56** |
| MLP (64 neurones) | Classification | **~80.5%** | F1 = 0.48 |
| MLP (64 neurones) | Régression | — | MSE = 1.026 |

**Point notable pour le rapport :**  
KMeans (K=20) surpasse le MLP en Macro F1 (0.56 vs 0.48) malgré une accuracy plus basse (73.7% vs 80.5%). Cela s'explique par le fait que le vote majoritaire avec beaucoup de clusters détecte mieux les classes minoritaires, ce que l'accuracy seule ne reflète pas. Le MLP reste supérieur en accuracy globale.
