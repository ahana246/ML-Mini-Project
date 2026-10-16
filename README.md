# Political News Bias Classification

A machine learning project that classifies U.S. political news articles as **Democrat-leaning or Republican-leaning** using article text. 

## Overview

Political bias classification is a text-classification task in which a model attempts to identify the political leaning associated with a news article.

The project uses TF-IDF text representations with Logistic Regression and Linear Support Vector Classification (Linear SVM). It is a reimplementation of the problem studied in the base project, *Categorizing News Articles on Political Bias* (Naive Bayes and SVM trained on congressional tweets and tested on news articles, which generalized poorly). Here the models are trained directly on labelled news articles instead.

## Objectives

- Build a simple, fast and interpretable political bias classifier using article text.
- Compare Logistic Regression and Linear SVM on identical features.
- Evaluate with a publisher-grouped split so that no outlet appears in both training and test sets.
- Analyze class-level performance using precision, recall, F1-score and a confusion matrix.
- Inspect the highest-weight terms to understand what drives predictions.
- Identify limitations in generalizing across publishers.

## Dataset

The project uses `bias_clean.csv`, containing **10,832 news articles** from roughly 300 media outlets.

The dataset includes the following fields:

| Column | Description |
|---|---|
| `page_text` | Main article text |
| `title` | Article headline |
| `site` | Publisher or media outlet (used to group the split) |
| `bias` | Original political bias label |
| `topic` | Article topic (not used by the model) |
| `date` | Publication date (not used by the model) |
| `url` | Article URL (used to drop duplicates) |

### Label mapping

The original five labels are consolidated into two classes, and `center` is excluded:

| Original labels | Final class | Original articles |
|---|---|---:|
| `left` | Democrat-leaning | 1,930 |
| `leaning-left` | Democrat-leaning | 2,951 |
| `center` | *Excluded* | 1,478 |
| `leaning-right` | Republican-leaning | 1,487 |
| `right` | Republican-leaning | 2,986 |

After mapping and filtering (duplicate URLs and very short texts are dropped), the prepared dataset has **9,354 articles from 193 publishers**:

| Class | Articles |
|---|---:|
| Democrat-leaning | 4,881 (52.2%) |
| Republican-leaning | 4,473 (47.8%) |

The labels are ideological proxies for the publisher's general lean. They approximate an article's orientation and do not show that it endorses a party.

## Methodology

### 1. Text preprocessing

The model input is the article title joined to its body text. Preprocessing includes label mapping, dropping duplicate URLs and very short texts, and text cleaning: lowercasing and removal of URLs, e-mail addresses, HTML tags and Twitter picture links, while keeping political vocabulary intact.

### 2. Feature extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts article text into numerical features, using unigrams and bigrams, English stop-word removal, `min_df=2`, `max_df=0.98`, at most 50,000 features and sublinear term frequency.

### 3. Models

**Logistic Regression**

A linear probabilistic classifier (`C=2`) with balanced class weights. It also provides class probabilities for new articles.

**Linear SVM**

A linear support vector classifier (`C=1`) with balanced class weights, trained on identical TF-IDF features so the comparison is fair.

### 4. Evaluation strategy

**Publisher-grouped split**

Publishers are separated between training and testing (`GroupShuffleSplit`, 20% of publishers held out, `random_state=42`). This gives 7,561 training articles from 154 publishers and 1,793 test articles from 39 publishers, with zero publisher overlap.

Articles from the same outlet share style and vocabulary, so a random article split would let a model benefit from publisher-specific patterns. The grouped split is a more demanding test of whether the model transfers to outlets it has never seen.

The test set is imbalanced: 1,272 Democrat-leaning (about 71%) and 521 Republican-leaning articles.

## Results

### Model comparison on held-out publishers

| Setup | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Project report (notebook) | Logistic Regression | 0.7948 | 0.746 |
| | Linear SVM | 0.7529 | 0.698 |
| Scripts re-run, `--legacy-cleaning`* | Logistic Regression | 0.7953 | 0.747 |
| | Linear SVM | 0.7563 | 0.702 |
| Scripts, corrected cleaning (default)* | Logistic Regression | 0.7842 | 0.736 |
| | Linear SVM | 0.7596 | 0.707 |

\* Re-run with scikit-learn 1.8.0 and pandas 3.0. Data preparation and the split reproduce the report exactly. Scores differ from the report by a few articles (library-version numerics). Differences between the legacy and corrected cleaning are within what a single split can resolve.

### Best model (Logistic Regression), per class

| Class | Precision | Recall | F1 | Articles |
|---|---:|---:|---:|---:|
| Democrat-leaning | 0.85 | 0.87 | 0.86 | 1,272 |
| Republican-leaning | 0.66 | 0.61 | 0.63 | 521 |

### Interpretation

- Logistic Regression outperformed Linear SVM on the held-out publishers (79.5% accuracy and 0.746 macro F1 vs 75.3% and 0.698). A simple TF-IDF and linear pipeline is a usable baseline without deep learning.
- Accuracy overstates performance. About 71% of the test set is Democrat-leaning, so always predicting that class would score roughly 71%; the model is about 8.5 points above that.
- Republican-leaning F1 (0.63) is well below Democrat-leaning F1 (0.86), so macro F1 is the fairer headline figure.
- The model leans partly on outlet cues. Top Republican-leaning terms include "fox", "fox news", "washington examiner", "click" and "news app"; top Democrat-leaning terms include "cnn", "ap", "nbc news", "abc news" and "story updated". Genuine political phrases such as "president trump" and "donald trump" also appear.

The results do not establish that the model has learned political bias independently of publisher-specific language. They come from a single split with a single random seed.

## Evaluation metrics

The project uses:

- **Accuracy:** Fraction of correctly classified articles.
- **Precision:** Fraction of predictions for a class that are correct.
- **Recall:** Fraction of actual articles in a class that the model identifies.
- **Macro F1:** Average F1-score across the two classes, weighting each class equally.
- **Confusion matrix:** Shows which classes are confused with one another.

Macro F1 is especially useful here because the test set contains far more Democrat-leaning than Republican-leaning articles.

## Project structure

```text
political-news-bias-classification/
├── README.md
├── requirements.txt               # Core dependencies (scripts)
├── requirements-dev.txt           # Adds Jupyter and seaborn for the notebook
├── data/
│   └── bias_clean.csv             # Local dataset; not committed
├── notebooks/
│   └── political_news_bias.ipynb  # Original exploratory notebook
├── src/
│   ├── data.py                    # Loading, label mapping and text cleaning
│   ├── train.py                   # Train and compare models, write artifacts
│   └── predict.py                 # Classify a new article
├── docs/
│   └── project_summary.pdf        # Written summary of the project
└── artifacts/                     # Generated outputs; not committed
    ├── political_bias_model.joblib
    ├── model_metadata.json
    ├── metrics.json
    ├── confusion_matrix.png
    └── model_comparison.png
```

## Installation

### Requirements

- Python 3.9 or newer (the notebook was run on Python 3.11)
- pip
- Git

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd political-news-bias-classification
```

Replace `<YOUR_REPOSITORY_URL>` with the URL of your GitHub repository.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

To also run the notebook, use `pip install -r requirements-dev.txt` instead.

### 3. Add the dataset

Place `bias_clean.csv` in the `data/` folder. The expected columns are described in [`data/README.md`](data/README.md).

## Running the experiments

Train and compare the models:

```bash
python -m src.train --data data/bias_clean.csv
```

This prepares the data, makes the publisher-grouped split, trains both models, prints the comparison and a classification report for the best model, and writes the model, metrics, top terms and figures to `artifacts/`.

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/bias_clean.csv` | Path to the CSV |
| `--output-dir` | `artifacts` | Where outputs are written |
| `--test-size` | `0.20` | Fraction of *publishers* held out |
| `--random-state` | `42` | Seed for the split and models |
| `--select-by` | `accuracy` | `accuracy` (as in the notebook) or `macro_f1` |
| `--legacy-cleaning` | off | Use the notebook's original, partly broken cleaning patterns |

If a split ends up with only one class in train or test, the script stops with a message; re-run with another `--random-state` (for example 7 or 21). Choose your evaluation method up front rather than changing the seed until the score looks good.

Reproduce the numbers in the project report:

```bash
python -m src.train --data data/bias_clean.csv --legacy-cleaning
```

Classify a new article with the saved model:

```bash
python -m src.predict --text "Lawmakers debated the new tax bill and its effect on jobs..."
python -m src.predict --file article.txt
```

Logistic Regression prints class probabilities. If the Linear SVM is selected as best, a decision score is shown instead, since it has no calibrated probabilities.

Run the original notebook:

```bash
jupyter lab notebooks/political_news_bias.ipynb
```

The notebook reads `bias_clean.csv` relative to its own folder: change `DATA_PATH` in the second code cell to `"../data/bias_clean.csv"` or copy the CSV next to the notebook.


## References

- Crain, D. M. and Erhan, Y. *Categorizing News Articles on Political Bias* (final write-up), the base project this work reimplements.

## Publishing to GitHub

```bash
git init -b main
git add .
git status            # check: data/bias_clean.csv and artifacts/* must NOT be listed
git commit -m "Initial commit: political news bias classification"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
