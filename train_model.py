"""Train the intent classifier and save metrics.

Trains a TF-IDF + LogisticRegression classifier on the sample utterances
in data/intents.csv (13 intents: greeting, smalltalk, admission deadlines,
merit, entry test, fees, scholarships, programs, documents, hostel,
eligibility, campus life, goodbye).

Run once before app.py:  python train_model.py
Outputs model.pkl and metrics.json (both gitignored).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "intents.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["utterance"].astype(str)
    y = df["intent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=100)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_intents": int(y.nunique()),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Trained on {metrics['n_train']} rows, tested on {metrics['n_test']}")
    print(f"Intents: {metrics['n_intents']}")
    print(f"Accuracy (held-out): {metrics['accuracy']}")
    print(f"Macro F1 (held-out): {metrics['macro_f1']}")
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
