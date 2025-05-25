"""
evaluate_model.py
Load test split & saved artefacts, produce confusion matrix, etc.
Extend with cross-validation, error analysis, or alternative metrics.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from utils import load_raw_dataset, load_artifact
from train_model import MODEL_DIR

def main() -> None:
    df = load_raw_dataset()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    X, y = df["message"], df["label"]

    # Split data the same way as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Re-use vectorizer & model
    vec = load_artifact(MODEL_DIR / "vectorizer.pkl")
    clf = load_artifact(MODEL_DIR / "spam_classifier.pkl")

    # Evaluate on TEST data only
    X_test_vec = vec.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print("=== TEST SET EVALUATION ===")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Ham', 'Spam'])
    disp.plot()
    plt.title('Confusion Matrix - Test Set')
    plt.show()

    # Show misclassified examples
    print("\n=== MISCLASSIFIED SPAM (False Negatives) ===")
    false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
    for i, msg in enumerate(false_negatives.head(5)):
        print(f"{i+1}. {msg[:100]}...")

if __name__ == "__main__":
    main()
