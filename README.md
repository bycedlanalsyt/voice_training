# 🎤 Projet Personnel - Reconnaissance Vocale avec Whisper

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-Personal-purple.svg)](LICENSE)

## 📋 Description du Projet

Ce projet explore les capacités du modèle **Whisper** d'OpenAI pour la reconnaissance vocale automatique en français. L'objectif est d'évaluer les performances du modèle face à différents types d'audios : propre, bruité et accentué, en utilisant la métrique **WER (Word Error Rate)**.

### 🎯 Objectifs

- **Évaluer** les performances du modèle Whisper sur différents types d'audios
- **Analyser** l'impact du bruit et des accents sur la qualité de transcription
- **Développer** un pipeline complet de traitement audio
- **Fournir** des métriques quantitatives d'évaluation

## 🏗️ Architecture du Projet

```
projet_machine_learning2/
├── 📁 notebooks/                 # Notebooks Jupyter du pipeline
│   ├── 01_data_prep.ipynb       # Préparation et exploration des données
│   ├── 02_transcription.ipynb    # Transcription automatique avec Whisper
│   └── 03_evaluation.ipynb      # Évaluation des performances
├── 📁 data/                     # Données audio
│   ├── raw/                     # Fichiers audio bruts
│   └── processed/               # Fichiers audio traités
├── 📁 reports/                  # Rapports et visualisations
│   ├── Rapport_Final_ML2_Whisper.pdf
│   ├── wer_by_category.png
│   ├── processing_time.png
│   └── file_distribution.png
├── 📄 requirements.txt          # Dépendances Python
├── 📄 generate_report.py        # Script de génération du rapport
├── 📄 .gitignore               # Fichiers à ignorer par Git
└── 📄 README.md                # Documentation du projet
```

## 🚀 Installation et Configuration

### Prérequis

- **Python** 3.8 ou supérieur
- **pip** ou **conda** pour la gestion des packages
- **Git** pour le contrôle de version

### Installation des Dépendances

1. **Cloner le repository** :
   ```bash
   git clone <url-du-repository>
   cd projet_machine_learning2
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Installer Whisper** (si nécessaire) :
   ```bash
   pip install openai-whisper
   ```

### Dépendances Principales

- `openai-whisper` : Modèle de reconnaissance vocale
- `pandas` : Manipulation et analyse de données
- `matplotlib` : Visualisation des données
- `numpy` : Calculs numériques
- `reportlab` : Génération de rapports PDF
- `jupyter` : Environnement de développement interactif

## 📊 Pipeline de Traitement

### 1. Préparation des Données (`01_data_prep.ipynb`)

- **Chargement** des fichiers audio bruts
- **Écoute et validation** de la qualité audio
- **Visualisation** des caractéristiques temporelles et fréquentielles
- **Nettoyage** et normalisation des fichiers
- **Création du manifest** pour le pipeline

### 2. Transcription Automatique (`02_transcription.ipynb`)

- **Configuration** du modèle Whisper
- **Chargement** des fichiers préparés
- **Transcription** automatique de chaque fichier
- **Sauvegarde** des résultats au format CSV
- **Visualisation** d'exemples de transcriptions

### 3. Évaluation des Performances (`03_evaluation.ipynb`)

- **Chargement** des transcriptions générées
- **Calcul du WER** pour chaque fichier
- **Analyse par catégorie** (propre, bruité, accentué)
- **Visualisation** des résultats
- **Discussion** des limites et améliorations

## 📈 Résultats Principaux

### Performances par Catégorie

| Catégorie | WER (%) | Nb Fichiers | Temps Moyen (s) |
|-----------|---------|-------------|-----------------|
| Propre | 12% | 15 | 2.3 |
| Bruit Faible | 25% | 12 | 2.8 |
| Bruit Fort | 40% | 10 | 3.2 |
| Accent | 55% | 8 | 3.5 |

### Observations Clés

- ✅ **Meilleures performances** sur les audios propres (12% WER)
- ⚠️ **Dégradation progressive** avec l'augmentation du bruit
- 🔴 **Défi majeur** avec les accents régionaux (55% WER)
- ⏱️ **Temps de traitement** proportionnel à la complexité

## 📋 Utilisation

### Exécution des Notebooks

1. **Lancer Jupyter** :
   ```bash
   jupyter notebook
   ```

2. **Exécuter dans l'ordre** :
   - `01_data_prep.ipynb`
   - `02_transcription.ipynb`
   - `03_evaluation.ipynb`

### Génération du Rapport PDF

```bash
python generate_report.py
```

Le rapport complet sera généré dans le dossier `reports/` avec :
- Analyse détaillée des résultats
- Graphiques et visualisations
- Recommandations et pistes d'amélioration

## 🔧 Configuration

### Structure des Données

Placez vos fichiers audio dans le dossier `data/raw/` organisés par catégorie :
```
data/raw/
├── propre/
├── bruit_faible/
├── bruit_fort/
└── accent/
```

### Paramètres Whisper

Le modèle Whisper peut être configuré dans `02_transcription.ipynb` :
- **Taille du modèle** : `tiny`, `base`, `small`, `medium`, `large`
- **Langue** : `fr` (français)
- **Format de sortie** : CSV avec métadonnées

## 📊 Métriques d'Évaluation

### WER (Word Error Rate)

Formule : `WER = (S + D + I) / N`

- **S** : Substitutions (mots incorrects)
- **D** : Délétions (mots manqués)
- **I** : Insertions (mots ajoutés)
- **N** : Nombre total de mots de référence

### Autres Métriques

- **Temps de traitement** par fichier
- **Taux de réussite** par catégorie
- **Analyse qualitative** des erreurs

## 🚧 Limitations et Améliorations

### Limitations Identifiées

- **Taille du dataset** limitée par catégorie
- **Variété des accents** restreinte
- **Conditions d'enregistrement** variables
- **Métriques** limitées au WER

### Pistes d'Amélioration

- 🔧 **Fine-tuning** sur corpus français étendu
- 🎵 **Préprocessing** avancé de réduction de bruit
- 📝 **Post-processing** avec règles linguistiques
- 📊 **Métriques supplémentaires** (BLEU, ROUGE)
- 📈 **Extension du dataset** avec plus de variétés

## 👨‍💻 Auteur

**Cédric BOIMIN**  
Développeur Python & Machine Learning

## 📄 Licence

Ce projet personnel est développé à des fins d'apprentissage et de démonstration.  
© 2025 - Projet personnel

## 🤝 Contribution

Les contributions et suggestions sont les bienvenues !  
N'hésitez pas à ouvrir une issue ou proposer des améliorations.

## 📚 Références

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Documentation Whisper](https://openai.com/research/whisper)
- [Métriques d'évaluation ASR](https://en.wikipedia.org/wiki/Word_error_rate)
- [Jupyter Notebook](https://jupyter.org/)

---

*Dernière mise à jour : Octobre 2025*
