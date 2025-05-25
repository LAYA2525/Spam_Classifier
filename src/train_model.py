"""
train_model.py
Train a TF-IDF + Multinomial NB spam/ham classifier and persist artefacts.

Run:
    python -m src.train_model
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from utils import load_raw_dataset, save_artifact

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def main() -> None:
    # 1  Load & prepare data
    df: pd.DataFrame = load_raw_dataset()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 2  Build pipeline with better parameters for spam detection
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                max_features=10000,  # Increased for more spam keywords
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for spam phrases
                min_df=1,  # Don't ignore rare spam words
                max_df=0.8,  # Lower threshold for common words
                sublinear_tf=True,  # Better for spam detection
                lowercase=True,
                token_pattern=r'\b\w+\b'  # Include single characters
            )),
            ("nb", MultinomialNB(alpha=0.01)),  # Much lower alpha for aggressive spam detection
        ]
    )

    # 3  Train
    pipeline.fit(X_train, y_train)

    # 4  Evaluate (quick peek – write a fuller report in evaluate_model.py)
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # 5  Persist artefacts - Save the entire pipeline AND components
    MODEL_DIR.mkdir(exist_ok=True)
    save_artifact(pipeline, MODEL_DIR / "spam_pipeline.pkl")  # Complete pipeline
    save_artifact(pipeline.named_steps["nb"], MODEL_DIR / "spam_classifier.pkl")
    save_artifact(pipeline.named_steps["tfidf"], MODEL_DIR / "vectorizer.pkl")


if __name__ == "__main__":
    main()
