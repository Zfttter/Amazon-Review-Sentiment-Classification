# Amazon Review Sentiment Classification

Sentiment classification on **All_Beauty** reviews from [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) using traditional machine learning methods (Complement Naive Bayes, Logistic Regression, Linear SVM) with TF-IDF features.

Two experiments are included:
- **Binary classification** (`main.py`): ratings 1–3 -> negative (0), 4–5 -> positive (1)
- **Ternary classification** (`main_ternary.py`): ratings 1–2 -> negative (0), 3 -> neutral (1), 4–5 -> positive (2), with error analysis and class-weight comparison

## Setup

On macOS, use **`python3`** / **`pip3`** (there is often no `python` / `pip` command).

```bash
cd amazon-sentiment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip` is still missing after `activate`, run: `python3 -m pip install -r requirements.txt`

## Run

```bash
# Binary classification
python main.py

# Ternary classification + error analysis
python main_ternary.py
```

The first run downloads the dataset from HuggingFace (may take several minutes). Subsequent runs use the local cache. Trained pipelines are saved under `results/models/`, plots under `results/figures/`.

**Note:** `datasets` is pinned to `<4` because HuggingFace `datasets` 4.x no longer executes Hub loading scripts; the code uses `load_dataset(..., trust_remote_code=True)`.

## Layout

```text
amazon-sentiment/
├── main.py               # Binary classification pipeline
├── main_ternary.py       # Ternary classification pipeline + error analysis
├── src/
│   ├── data_loader.py    # HuggingFace loading, cleaning, sampling, label mapping
│   ├── preprocessing.py  # clean_text(), TF-IDF Pipeline builder
│   ├── models.py         # Model definitions (binary + ternary, with/without class weights)
│   └── evaluation.py     # Confusion matrices, ROC curves, model comparison,
│                         # error analysis, class-weight comparison plot
```
