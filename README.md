# Cross-Domain Visual Robustness

A computer-vision robustness project for studying how classical machine-learning models trained on one visual domain degrade when evaluated on a different visual domain. The project uses MNIST-to-SVHN transfer as a compact testbed for domain shift, feature robustness, and adaptation baselines.

The repository is GitHub-ready without raw datasets. MNIST and SVHN are downloaded locally at runtime through `torchvision` when the `--download` flag is used.

## Research Motivation

Models often perform well when train and test data come from the same distribution, but fail sharply when the visual domain changes. MNIST and SVHN contain digit labels, yet they differ strongly in background, color, texture, clutter, and acquisition conditions. This project measures that gap directly by training on MNIST and testing on both MNIST and SVHN.

The goal is to compare how different feature representations behave under domain shift:

- raw resized pixel vectors
- HOG descriptors
- PCA-compressed pixel features
- classical classifiers trained on the source domain
- few-shot target-domain adaptation and simple ensemble baselines

## Pipeline

```text
MNIST source-domain digits
        |
        v
Feature extraction: raw pixels, HOG, PCA
        |
        v
Train classical classifiers on MNIST
        |
        v
Evaluate on MNIST test set and SVHN target-domain set
        |
        v
Measure accuracy degradation under domain shift
        |
        v
Generate confusion matrix, t-SNE, feature-distribution, and summary tables
```

## Models Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline over visual features |
| SVM | Nonlinear margin-based classifier |
| Random Forest | Tree-ensemble baseline |
| kNN | Local-neighborhood classifier |
| Gaussian Naive Bayes | Probabilistic baseline |

## Generated Outputs

Running the script creates:

```text
outputs/
├── plots/
│   ├── domain_transfer_accuracy.png
│   ├── svhn_confusion_random_forest_hog.png
│   ├── svhn_hog_tsne.png
│   └── hog_feature_distribution.png
└── tables/
    ├── cross_domain_accuracy_summary.csv
    ├── adaptation_baselines.csv
    └── experiment_metadata.csv
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── cross_domain_visual_robustness.py
```

Raw datasets are excluded from the repository and ignored by `.gitignore`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment and download datasets locally if missing:

```bash
python cross_domain_visual_robustness.py --download
```

If the datasets already exist locally under `data/`, run:

```bash
python cross_domain_visual_robustness.py
```

## Why This Project Matters

This project demonstrates domain-shift evaluation, feature robustness analysis, cross-domain degradation measurement, and lightweight adaptation baselines. It is useful for explaining why model performance on clean benchmark data does not always transfer to visually different real-world data.
