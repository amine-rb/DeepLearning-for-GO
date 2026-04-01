# Go 19×19 — Comparaison d'architectures sous contrainte de 100k paramètres

> Projet académique — Master 2 IASD, Université Paris-Dauphine PSL (2025–2026)  
> Amine ROUIBI · Thomas SINAPI · Mohamed ZOUAD

---

## Vue d'ensemble

Ce projet compare **14 architectures de réseaux de neurones profonds** pour la prédiction de coups et l'évaluation de position dans le jeu de Go 19×19, sous une contrainte stricte de **100 000 paramètres maximum**.

La tâche est une prédiction à deux têtes :
- **Tête politique** : distribution de probabilité sur les 361 intersections du plateau (softmax)
- **Tête valeur** : probabilité de victoire du joueur courant (sigmoïde scalaire)

Le meilleur modèle atteint **44,56 % de précision politique** en validation — soit +43,99 pp par rapport au baseline MLP.

---

## Résultats

| Rang | Modèle | Phase | Données | Acc. politique |
|------|--------|-------|---------|----------------|
| 🥇 1 | M18 – EfficientFormer-Go | 3 | 50k | **44,56 %** |
| 🥈 2 | M11 – MBConv + SE | 3 | 50k | 44,39 % |
| 🥉 3 | M12 – Inverted + ECA | 3 | 50k | 43,11 % |
| 4 | M15 – ConvNeXt-Lite† | 3 | 50k | 41,19 % |
| 5 | M4 – ResNet Tiny | 2 (retrain) | 50k | 35,29 % |
| 6 | M9 – ResNet + Dilated | 2 (retrain) | 50k | 35,01 % |
| 7 | M3 – CNN + BatchNorm | 2 (retrain) | 50k | 33,87 % |
| … | M1 – MLP Baseline | 1 | 10k | 0,57 % |

†M15 non convergé à la fin du run (courbe encore croissante).

---

## Structure du projet

```
├── rapport_architectures.md   # Code complet des 14 architectures (M1–M18)
├── GO_report.pdf              # Rapport complet (15 pages)
└── README.md
```

---

## Les 14 architectures

### Phase 1 — Exploration (N = 10 000 positions)

| Modèle | Architecture | Params | Acc. |
|--------|-------------|--------|------|
| M1 | MLP Baseline | 99 904 | 0,57 % |
| M2 | CNN Shallow | 97 388 | 11,12 % |
| M3 | CNN + BatchNorm | 99 752 | 29,87 % |
| M4 | ResNet Tiny | 99 628 | 31,08 % |
| M5 | Depthwise Separable CNN | 99 752 | 0,39 % |
| M6 | CNN Asymétrique | 99 752 | 6,13 % |
| M7 | MobileNet-style | 99 752 | 12,19 % |
| M8 | ResNet + SE | 99 628 | 11,41 % |
| M9 | ResNet + Dilated Conv | 99 628 | **32,71 %** |
| M10 | CNN + Transformer | 99 498 | 0,39 % |

### Phase 2 — Réentraînement des top 3 (N = 50 000 positions)

| Modèle | Phase 1 | Phase 2 | Gain |
|--------|---------|---------|------|
| M4 – ResNet Tiny | 31,08 % | **35,29 %** | +4,21 pp |
| M9 – ResNet + Dilated | 32,71 % | 35,01 % | +2,30 pp |
| M3 – CNN + BatchNorm | 29,87 % | 33,87 % | +4,00 pp |

### Phase 3 — Architectures efficientes modernes (N = 50 000 positions)

| Modèle | Architecture | Attention | Params | Acc. |
|--------|-------------|-----------|--------|------|
| M11 | MBConv + SE | SE | 98 242 | 44,39 % |
| M12 | Inverted Residual + ECA | ECA (≤5 params) | 93 773 | 43,11 % |
| M15 | ConvNeXt-Lite | — | 92 863 | 41,19 %† |
| M18 | EfficientFormer-Go | MHA poolé | 99 666 | **44,56 %** |

---

## Protocole expérimental

Tous les modèles partagent **exactement les mêmes hyperparamètres** — seule l'architecture varie.

| Hyperparamètre | Valeur |
|---------------|--------|
| Optimiseur | Adam |
| Learning rate | 0,001 |
| Régularisation L2 | 1e-4 |
| Batch size | 128 × #GPU |
| Précision | Mixed float16 |
| Early stopping (patience) | 20 époques |
| ReduceLROnPlateau | facteur 0,5, patience 10 |
| Graine aléatoire | 42 (`TF_DETERMINISTIC_OPS=1`) |
| Epochs max | 300 |

**Infrastructure** : Kaggle Notebooks (2×T4 / P100), MirroredStrategy multi-GPU, suivi W&B, checkpoints sauvegardés sur Google Drive.

---

## Installation et reproduction

```bash
# Données (module golois — Dauphine)
wget https://www.lamsade.dauphine.fr/~cazenave/project2026.zip
unzip project2026.zip

# Dépendances
pip install tensorflow wandb
```

```python
import wandb
wandb.login(key="VOTRE_CLE")

# Lancer un modèle (exemple M18)
history, best_loss, best_acc = train_model(build_model, config)
```

Le pipeline `train_model()` gère automatiquement : early stopping, checkpointing, logging W&B, export CSV et synchronisation Drive.

---

## Conclusions principales

1. **Le biais inductif spatial est indispensable** : le MLP (0,57 %) et le CNN dépthwise-separable (0,39 %) échouent sans structure convolutionnelle adaptée.
2. **Le volume de données est un levier secondaire** : ×5 données apporte +2 à +4 pp sur les mêmes architectures, mais ne suffit pas à dépasser le plafond des ResNet (~35 %).
3. **Les architectures efficientes modernes dominent** : MBConv, ECA, ConvNeXt et EfficientFormer apportent +9 pp supplémentaires à budget paramétrique identique.

---

## Références

- [AlphaGo] Silver et al., *Nature* 2016
- [MobileNetV1] Howard et al., arXiv 2017
- [MobileNetV2] Sandler et al., CVPR 2018
- [SE-Net] Hu et al., CVPR 2018
- [ECA-Net] Wang et al., CVPR 2020
- [EfficientNet] Tan & Le, ICML 2019
- [ConvNeXt] Liu et al., CVPR 2022
- [EfficientFormer] Li et al., NeurIPS 2022

---

*Cours Apprentissage Profond — M2 IASD — Université Paris-Dauphine PSL — 2025/2026*
