# Housing Market Intelligence — Iteration 5 Dataset

This version uses one dataset: `housing_iteration_5_classification.csv`.

The notebook combines unsupervised and supervised ML:

- clustering market archetypes without using the `Expensive` label,
- comparing discovered archetypes with the expensive-house target,
- training supervised classifiers,
- evaluating ROC-AUC, confusion matrix, thresholds and feature importance.

## Run

Open `notebooks/housing_market_intelligence_iteration5_colab.ipynb` in Google Colab and run all cells.

## Structure

```text
housing_market_intelligence_ITERATION5/
├── data/
│   └── housing_iteration_5_classification.csv
├── images/
└── notebooks/
```
