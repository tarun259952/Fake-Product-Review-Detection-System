# 🕵️ Fake Product Review Detection System

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-Dashboard-3F4F75?logo=plotly&logoColor=white)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-87.20%25-brightgreen)

An end-to-end NLP and machine learning pipeline that classifies e-commerce product reviews as **genuine (`OR`)** or **computer-generated / fake (`CG`)**. The project covers text preprocessing, feature engineering, a six-model benchmark, model persistence, and an interactive Plotly Dash dashboard for analytics and live prediction.

---

## 📑 Table of Contents

1. [Objective](#-objective)
2. [Project Structure](#-project-structure)
3. [Key Results](#-key-results)
4. [Dataset Overview](#-dataset-overview)
5. [Methodology](#-methodology)
6. [Model Performance](#-model-performance)
7. [Analysis](#-analysis)
8. [Conclusion & Recommendation](#-conclusion--recommendation)
9. [Getting Started](#-getting-started)
10. [Limitations & Future Work](#-limitations--future-work)
11. [Tech Stack](#-tech-stack)

---

## 🎯 Objective

Fake and AI-generated reviews distort purchase decisions and erode trust in online marketplaces. This project builds a supervised text-classification system that:

- Analyzes product review text using NLP techniques
- Identifies linguistic and behavioral patterns associated with fake reviews
- Benchmarks multiple classification algorithms to find the most reliable detector
- Surfaces the results through an interactive dashboard for exploration and live prediction

**Target variable:** `CG` (fake / computer-generated) vs. `OR` (original / genuine)

---

## 📁 Project Structure

```
├── Data Loading, EDA & Text Preprocessing.ipynb     # Data ingestion, cleaning, exploratory analysis
├── Feature Engineering and model comparisn.ipynb    # NLP feature extraction + model training/comparison
├── Model saving and advance evaluation.ipynb        # Model persistence (joblib) & detailed evaluation
├── fake_review_dashboard.ipynb                      # Two-tab Plotly Dash app (analytics + live prediction)
├── models/                                          # Saved scikit-learn pipelines (created by the model-saving notebook)
│   ├── best_model_pipeline.pkl
│   ├── support_vector_machine_pipeline.pkl
│   ├── logistic_regression_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   ├── naive_bayes_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   └── knn_k5_pipeline.pkl
└── images/                                          # Charts used in this README
```

> ⚠️ `random_forest_pipeline.pkl` (~131 MB) exceeds GitHub's 100 MB file limit. Add it to `.gitignore`, track it with [Git LFS](https://git-lfs.com/), or regenerate it by running the model-saving notebook.

---

## 🏆 Key Results

| Metric | Best Model | Score |
|---|---|---|
| Accuracy | Support Vector Machine | **87.20%** |
| F1-Score | Support Vector Machine | **87.24%** |
| Precision | Logistic Regression | **88.13%** |
| Fewest false accusations (genuine flagged as fake) | Logistic Regression | **809** of 7,076 genuine reviews (11.4%) |
| Most fakes caught (Recall) | KNN (k=5) | 95.27% — but unusable, see [Analysis](#-analysis) |

> **Takeaway:** SVM and Logistic Regression are the strongest, most deployment-ready models. SVM leads on overall accuracy/F1, while Logistic Regression is the safest choice when wrongly flagging a genuine reviewer is the costlier mistake.

---

## 📊 Dataset Overview

- **Test set:** 14,151 reviews, almost perfectly balanced — 7,075 `CG` vs. 7,076 `OR`. With no class imbalance to correct for, accuracy is a trustworthy headline metric here.

### Rating distribution

Ratings are heavily skewed positive: **5-star reviews make up 60.7%** of the data, followed by 4-star (19.7%). Ratings of 1–3 stars together account for under 20%. This skew is typical of e-commerce platforms, but it is also a known signature that fake-review campaigns exploit.

![Rating proportions](images/rating_distribution.png)

### Review length distribution

Review length is strongly right-skewed: most reviews are under ~250 characters, with a long tail beyond 1,400 characters. Split by label, `CG` reviews cluster tightly at the short end, while `OR` reviews spread much wider and make up most of the long tail. The two classes overlap heavily in the short range, so **length alone is not enough to separate them** — the models are likely leaning on lexical and stylistic features, with length as at most a supporting signal.

![Review length distribution](images/review_length_distribution.png)

---

## 🔬 Methodology

1. **Preprocessing** – Text cleaning, tokenization, and normalization of raw review text (NLTK).
2. **Feature Engineering** – NLP-based text vectorization feeding six classical ML algorithms.
3. **Model Training** – Six classifiers trained and evaluated on a held-out test split.
4. **Persistence** – Each model saved as a full scikit-learn pipeline via `joblib`, plus a `best_model_pipeline.pkl` used by the dashboard.
5. **Deployment** – Two-tab Plotly Dash dashboard: one tab for model analytics/comparison, one for live review prediction.

**Models benchmarked:** Support Vector Machine, Logistic Regression, Random Forest, Multinomial Naive Bayes, Decision Tree, K-Nearest Neighbors (k=5).

---

## 📈 Model Performance

`CG` (fake) is treated as the positive class, so **precision** = "when the model says fake, how often is it right?" and **recall** = "what share of fakes does it catch?"

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Support Vector Machine** | **87.20%** | 86.97% | 87.52% | **87.24%** |
| **Logistic Regression** | 86.74% | **88.13%** | 84.92% | 86.50% |
| Random Forest | 84.66% | 81.99% | 88.82% | 85.27% |
| Naive Bayes | 84.40% | 80.47% | 90.84% | 85.34% |
| Decision Tree | 75.10% | 75.17% | 74.94% | 75.06% |
| KNN (k=5) | 61.86% | 57.11% | **95.27%** | 71.41% |

![All model metrics](images/model_metrics_comparison.png)

### Error breakdown (14,151 test reviews)

| Model | Fakes caught (TP) | Fakes missed (FN) | Genuine kept (TN) | **Genuine wrongly flagged (FP)** | False-flag rate |
|---|---|---|---|---|---|
| Logistic Regression | 6,008 | 1,067 | 6,267 | **809** | 11.4% |
| Support Vector Machine | 6,192 | 883 | 6,148 | **928** | 13.1% |
| Random Forest | 6,284 | 791 | 5,696 | **1,380** | 19.5% |
| Naive Bayes | 6,427 | 648 | 5,516 | **1,560** | 22.0% |
| Decision Tree | 5,302 | 1,773 | 5,325 | **1,751** | 24.7% |
| KNN (k=5) | 6,740 | 335 | 2,014 | **5,062** | 71.5% |

*False-flag rate = FP ÷ 7,076 genuine reviews.*

![Confusion matrices](images/confusion_matrices.png)

### Earlier accuracy comparison

An earlier accuracy comparison from the model-comparison notebook shows nearly identical results (within ~0.35 percentage points of the final evaluation above) and the same ranking: SVM 87.27% › Logistic Regression 86.67% › Random Forest 84.88% › Multinomial NB 84.06% › Decision Tree 75.22% › KNN 61.75%.

![Model accuracy comparison](images/model_accuracy_comparison.png)

---

## 🔍 Analysis

- **SVM is the best all-around performer.** It tops both accuracy (87.20%) and F1 (87.24%), with precision and recall closely balanced (~87%), so it doesn't heavily favor either class.
- **Logistic Regression is a very close second** and has the highest precision of any model (88.13%). It also makes the *fewest false accusations* (809), at the cost of missing slightly more fakes than SVM.
- **Random Forest and Naive Bayes sit in the same tier** (~84–85% accuracy) with a recall-over-precision skew: they catch more fakes (6,284 and 6,427) but wrongly flag noticeably more genuine reviews (1,380 and 1,560) than SVM or Logistic Regression.
- **KNN is misleading if you only look at recall.** Its 95.27% recall looks best on paper, but it flags **5,062 genuine reviews as fake — over 70% of all genuine reviews**. It effectively over-predicts "fake", which would penalize most legitimate reviewers in a real moderation system.
- **Decision Tree underperforms** (~75% across the board), consistent with a single tree's tendency to overfit without bagging or boosting.
- **Model size matters for deployment.** The saved SVM and Logistic Regression pipelines are each ~820 KB, versus ~131 MB for Random Forest (roughly 160× larger) and ~8.5 MB for KNN — and the small linear models are also the more accurate ones.

| Saved pipeline | Size |
|---|---|
| `support_vector_machine_pipeline.pkl` | 820 KB |
| `logistic_regression_pipeline.pkl` | 820 KB |
| `decision_tree_pipeline.pkl` | 1.1 MB |
| `naive_bayes_pipeline.pkl` | 1.5 MB |
| `knn_k5_pipeline.pkl` | 8.5 MB |
| `random_forest_pipeline.pkl` | ~131 MB |

**Ranking for production use:** SVM ≈ Logistic Regression > Random Forest ≈ Naive Bayes > Decision Tree > KNN

---

## ✅ Conclusion & Recommendation

Across all six algorithms, **Support Vector Machine and Logistic Regression are the most suitable for deployment**: they catch the most fakes *while* minimizing false accusations against genuine reviewers — a critical requirement for any system acting on live user-generated content. They are also lightweight and fast to load.

- Choose **SVM** for the best overall accuracy and F1.
- Choose **Logistic Regression** if wrongly flagging genuine reviewers is the bigger business risk (highest precision, fewest false positives).
- Naive Bayes and Random Forest are reasonable secondary options when maximizing recall matters more than avoiding false alarms.
- **KNN is not recommended** for production, and Decision Tree underperforms as a standalone model.

The positively skewed rating distribution (60.7% 5-star) underlines the real-world relevance of the problem, and the heavily overlapping length distributions suggest the models rely mostly on lexical and stylistic cues rather than surface metadata.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly dash joblib jupyter
```

### 2. Run the notebooks in order

1. `Data Loading, EDA & Text Preprocessing.ipynb`
2. `Feature Engineering and model comparisn.ipynb`
3. `Model saving and advance evaluation.ipynb` — creates the `.pkl` pipelines
4. `fake_review_dashboard.ipynb` — launches the dashboard (Dash serves locally, by default at `http://127.0.0.1:8050`)

### 3. Use a saved model directly

```python
import joblib

pipeline = joblib.load("models/best_model_pipeline.pkl")

reviews = [
    "Great product, works exactly as described and arrived on time.",
    "Best product ever buy now amazing quality highly recommend.",
]
print(pipeline.predict(reviews))  # predicted class label for each review (CG / OR)
```

> The pipelines bundle preprocessing and the classifier, so you can pass raw review text straight in. Load them with the **same scikit-learn version** used to train them, or pickle compatibility errors may occur.

---

## 🚧 Limitations & Future Work

- **Single train/test split.** Results come from one held-out split; k-fold cross-validation would give more reliable estimates and confidence intervals.
- **Default hyperparameters.** Models were compared as-is; tuning (e.g. grid/random search for SVM `C`, Logistic Regression regularization, KNN `k`) could narrow the gaps.
- **Decision threshold.** Adjusting the classification threshold on Logistic Regression can trade precision against recall to match moderation policy.
- **Ensembling.** A stacking ensemble of SVM, Logistic Regression, and Random Forest may further reduce false positives.
- **Explainability.** Inspecting top-weighted terms of the linear models would confirm which lexical cues separate `CG` from `OR`.
- **Stronger text models.** Transformer-based classifiers (e.g. fine-tuned BERT variants) could capture style and fluency patterns that bag-of-words models miss.
- **Domain generalization.** Performance should be validated on reviews from other platforms and product categories before real-world use.

---

## 🧰 Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `NLTK` · `matplotlib` / `seaborn` · `Plotly Dash` · `joblib`
