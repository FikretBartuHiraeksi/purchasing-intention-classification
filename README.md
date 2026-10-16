# Online Shopper Purchase Intention Classification

Predicting whether an online shopping session ends in a purchase, using session-level behaviour data and gradient boosting in Python.

## Overview

Classifying whether a customer's visit ends in a purchase allows us to uncover the key drivers behind purchasing intent and identify actionable areas to optimize and boost sales conversion.

The full write-up with all charts is in 📄 [Full Project Report (PDF)](Customer%20Classifying.pdf) The code is in 📓 [Jupyter Notebook](customer-classifying.ipynb).

## Data

- **Dataset:** Online Shoppers Purchasing Intention Dataset (`online_shoppers.csv`), 12,330 sessions x 18 features, target `Revenue` (15.5% positive)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Authors:** C. O. Sakar, S. O. Polat, M. Katircioglu, Y. Kastro (2019)
- **License:** CC BY 4.0. The CSV in this repository is the original file, unmodified. All cleaning is done in the notebook.

## Method

1. **Diagnosis:** checked data types, nulls, duplicates, outliers and the target balance. Found 125 duplicate rows and 1,078 page/duration inconsistencies (pages visited with zero time, which looks like users closing the page quickly rather than a tracking error).
2. **Exploration:** purchase rate by month, visitor type, special day and traffic type; distributions of numeric features by outcome; Spearman correlation heatmap.
3. **Split:** dropped duplicates first, then a stratified 80/20 train/test split (9,764 / 2,441 sessions). The test set was not touched until the final evaluation.
4. **Preprocessing (inside pipelines):** StandardScaler on numeric features, one-hot encoding for categorical features, and log transform of skewed count and duration features for logistic regression.
5. **Model comparison:** 5-fold stratified cross-validation for a dummy baseline, logistic regression, random forest and histogram gradient boosting. Average precision was the main metric because the classes are imbalanced.
6. **Tuning:** `RandomizedSearchCV` (30 iterations) on the gradient boosting model.
7. **Leakage check:** compared the tuned model with and without `PageValues`.
8. **Threshold:** chose the decision threshold (0.722) from out-of-fold predictions on the training set, before opening the test set.
9. **Interpretation:** permutation importance and SHAP values.

## Results

Cross-validated on the training set (mean over 5 folds):

| Model | Precision | Recall | F1 | Average precision |
|---|---|---|---|---|
| Dummy (majority class) | 0.000 | 0.000 | 0.000 | 0.156 |
| Logistic regression | 0.545 | 0.796 | 0.647 | 0.663 |
| Random forest | 0.599 | 0.773 | 0.675 | 0.734 |
| Hist gradient boosting | 0.589 | 0.786 | 0.673 | 0.743 |
| Gradient boosting, tuned | 0.532 | 0.842 | 0.652 | 0.753 |

Final test set result (threshold 0.722): precision 0.647, recall 0.733, F1 0.687, average precision 0.755. The confusion matrix has 280 true positives, 102 false negatives, 153 false positives and 1,906 true negatives.

**Key finding:** `PageValues` dominates the model. Without it, cross-validated average precision falls from 0.753 to 0.378 and F1 from 0.652 to 0.419. Could be indicating a potantial target leakage.

## Limitations

- Feature PageValues is a very dominating feature. Scores with and without PageValues could be a indicator to a target leakage.
- Tuning improved cross-validated average precision from 0.743 (±0.019) to 0.753 (±0.018), a difference smaller than the fold-to-fold spread, so the gain is small.
- The data covers 10 months of the year (January and April are missing), and `SpecialDay` is non-zero only in February and May, so seasonal and holiday effects can't be separated.
- Test results come from a single split.

## Tools

- 🐍 Python 3
- 📓 Jupyter Notebook
- 🐼 pandas
- 🔢 numpy
- 🤖 scikit-learn
- 📊 seaborn
- 🔍 shap
- 🔬 scipy

## How to run

```
pip install -r requirements.txt
jupyter notebook customer-classifying.ipynb
```

## Author

Fikret Bartu Hıraekşi
