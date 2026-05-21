# 🏠 Housing Market Intelligence  
### Discovering market archetypes before predicting expensive homes

<p align="center">
  <img src="images/ml_pipeline_infographic.png" width="900">
</p>

---

# Overview

I started this project with a simple supervised learning question:

> **Can a machine learning model predict whether a house is expensive?**

But after exploring the dataset, the more interesting question became:

> **If I completely remove the target label, will the housing market still organize itself into meaningful groups?**

That transformed the project from a standard classification notebook into a combined:

- Unsupervised Learning
- Supervised Learning
- Market Segmentation
- Feature Engineering
- Model Interpretation
- ML Storytelling

workflow.

The final result is a **research-style machine learning investigation** that combines clustering, dimensionality reduction, predictive modeling and interpretability inside one reproducible Google Colab notebook.

---

# 🚀 What Makes This Project Different

Most housing ML notebooks stop after:

- training a classifier,
- printing accuracy,
- plotting feature importance.

This project goes further.

Instead of starting with labels, I first let the data organize itself into **market archetypes** using unsupervised learning.

Only afterward do I bring back the target variable and ask:

> **Do expensive houses naturally concentrate inside particular market segments?**

That creates a much stronger ML narrative than a standalone classifier.

---

# 🧠 Main Ideas Explored

---

## 1. Unsupervised Learning

The notebook discovers hidden structure in the housing market using:

- PCA
- MiniBatchKMeans
- Agglomerative Clustering
- DBSCAN diagnostics

The clustering stage creates interpretable housing archetypes such as:

- Premium Amenity Homes
- Compact Basic Homes
- Outdoor Space Homes
- Large Lot Homes

instead of generic cluster IDs.

---

## 2. Supervised Learning

After segmentation, the project predicts the `Expensive` target using:

- Logistic Regression
- Random Forest
- Gradient Boosting

The final Random Forest model achieves approximately:

```python
ROC-AUC ≈ 0.98
```

while remaining interpretable and reproducible.

---

## 3. Feature Engineering

The project includes extensive market-oriented feature engineering:

- quality signals,
- amenity signals,
- outdoor-space metrics,
- density ratios,
- market-strength indicators,
- safe engineered housing features.

The goal is not only prediction performance, but also interpretability.

---

## 4. Archetype Analysis

Clusters are profiled and interpreted using:

- quality indicators,
- lot size,
- basement area,
- garages,
- fireplaces,
- outdoor space,
- premium amenities.

The notebook uses **z-score heatmaps** instead of misleading min-max scaling so the archetypes are easier to interpret.

---

# 🔄 Why Combine Supervised and Unsupervised Learning?

<p align="center">
  <img src="images/supervised_unsupervised_bridge.png" width="900">
</p>

The two approaches answer different questions.

### Unsupervised learning asks:

> *What structure exists before I use labels?*

### Supervised learning asks:

> *Can the expensive-house label be predicted reliably?*

When both perspectives point in similar directions, the project becomes stronger than a normal ML notebook.

It becomes a small **market intelligence study**.

---

# ⚙️ Technical Highlights

## Mixed-Type Preprocessing

The notebook handles:

- numeric features,
- ordinal quality variables,
- nominal categorical variables,
- missing values,
- mixed-type preprocessing pipelines.

---

## Stable Clustering Pipeline

The clustering workflow was redesigned to be:

- faster,
- more stable,
- Colab-friendly,
- “Run All” safe.

Key improvements:

- clustering performed on PCA space,
- MiniBatchKMeans instead of slower full KMeans,
- automatic fallback for `best_k`,
- robust silhouette analysis,
- DBSCAN treated as diagnostic instead of overclaimed.

---

## Model Interpretation

The supervised section includes:

- ROC-AUC,
- confusion matrix,
- threshold analysis,
- feature importance analysis,
- cluster-to-target bridge analysis.

---

# 📂 Repository Structure

```text
housing-market-intelligence/
│
├── data/
│   └── housing_iteration_5_classification.csv
│
├── images/
│   ├── ml_pipeline_infographic.png
│   ├── supervised_unsupervised_bridge.png
│   └── github_project_map.png
│
├── notebooks/
│   └── housing_market_intelligence_iteration5_colab.ipynb
│
└── README.md
```

---

# ▶️ Running the Notebook

## Recommended Google Drive structure

```text
My Drive/
└── Colab Notebooks/
    ├── images/
    └── housing_iteration_5_classification.csv
```

---

## Google Colab

The notebook is fully designed for Google Colab.

For stable image rendering, it uses:

```python
from IPython.display import Image, display
```

instead of fragile markdown relative paths.

---

# 🛠️ Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Google Colab

---

# 🗺️ Project Map

<p align="center">
  <img src="images/github_project_map.png" width="900">
</p>

---

# 📌 Final Reflection

What I like most about this project is that it does not treat machine learning as only:

> *fit model → print accuracy*

Instead, the workflow investigates the housing market from multiple angles:

- discover hidden structure,
- build interpretable archetypes,
- connect clusters with labels,
- predict expensive properties,
- explain what drives predictions.

That makes the notebook feel closer to a real ML research workflow than a standard tutorial notebook.

---
