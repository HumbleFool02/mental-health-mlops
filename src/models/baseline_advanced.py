"""
Advanced Baseline Experiments
- Resampling techniques (SMOTE, Random Oversampling)
- Advanced models (XGBoost, LightGBM)
- Text preprocessing variations (stemming, lemmatization)
"""
import os
import sys
import time
import warnings

import mlflow
import mlflow.sklearn
import nltk
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.utils import set_seed

warnings.filterwarnings("ignore")


# Download NLTK resources
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_data():
    """Load train and validation data"""
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    return train_df, val_df


def preprocess_text(text, method="none"):
    """
    Apply text preprocessing

    Args:
        text: input text
        method: 'none', 'stem', 'lemma', 'no_stopwords'
    """
    if method == "none":
        return text

    words = text.split()

    if method == "stem":
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in words])

    elif method == "lemma":
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in words])

    elif method == "no_stopwords":
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in words if word not in stop_words])

    return text


def apply_preprocessing(texts, method="none"):
    """Apply preprocessing to all texts"""
    if method == "none":
        return texts

    print(f"Applying preprocessing: {method}...")
    return [preprocess_text(text, method) for text in texts]


def apply_resampling(X, y, method="none"):
    """
    Apply resampling technique

    Args:
        X: feature matrix
        y: labels
        method: 'none', 'smote', 'random_oversample', 'smote_tomek'
    """
    if method == "none":
        return X, y

    print(f"Applying resampling: {method}...")

    if method == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=3)
    elif method == "random_oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "smote_tomek":
        sampler = SMOTETomek(random_state=42)
    else:
        return X, y

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print(f"Original samples: {X.shape[0]}")
    print(f"Resampled samples: {X_resampled.shape[0]}")

    return X_resampled, y_resampled


def create_model(model_type="xgboost", num_classes=6):
    """Create advanced ML model"""

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=num_classes,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    elif model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_class=num_classes,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    elif model_type == "logistic":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
        )

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(y_true, y_pred, label_encoder):
    """Calculate evaluation metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    return metrics, class_report


def train_experiment(
    train_df,
    val_df,
    model_type="xgboost",
    preprocessing="none",
    resampling="none",
    ngram_range=(1, 2),
    max_features=5000,
    experiment_name="baseline-experiments-advanced",
):
    """
    Run advanced training experiment
    """

    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    run_name = f"{model_type}_{preprocessing}_{resampling}"
    with mlflow.start_run(run_name=run_name):
        print(f"\n{'=' * 60}")
        print(f"Experiment: {run_name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("preprocessing", preprocessing)
        mlflow.log_param("resampling", resampling)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("max_features", max_features)

        # Prepare data
        X_train = train_df["text"].values
        y_train = train_df["label"].values
        X_val = val_df["text"].values
        y_val = val_df["label"].values

        # Apply text preprocessing
        X_train = apply_preprocessing(X_train, preprocessing)
        X_val = apply_preprocessing(X_val, preprocessing)

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        num_classes = len(label_encoder.classes_)
        mlflow.log_param("num_classes", num_classes)

        # Vectorize
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        print(f"Feature matrix shape: {X_train_vec.shape}")
        mlflow.log_param("actual_features", X_train_vec.shape[1])

        # Apply resampling
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train_vec, y_train_encoded, resampling
        )
        mlflow.log_param("train_samples_after_resampling", len(y_train_resampled))

        # Create and train model
        print("Training model...")
        model = create_model(model_type, num_classes)
        model.fit(X_train_resampled, y_train_resampled)

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # Predictions
        print("Making predictions...")
        y_train_pred = model.predict(X_train_vec)
        y_val_pred = model.predict(X_val_vec)

        # Evaluate
        train_metrics, train_report = evaluate_model(
            y_train_encoded, y_train_pred, label_encoder
        )
        val_metrics, val_report = evaluate_model(
            y_val_encoded, y_val_pred, label_encoder
        )

        print("\nValidation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {val_metrics['f1_weighted']:.4f}")

        # Log metrics
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        # Log per-class metrics
        print("\nPer-Class F1 Scores (Validation):")
        for class_name in label_encoder.classes_:
            if class_name in val_report:
                f1 = val_report[class_name]["f1-score"]
                precision = val_report[class_name]["precision"]
                recall = val_report[class_name]["recall"]

                print(
                    f"  {class_name:20s}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}"
                )

                mlflow.log_metric(f"val_f1_{class_name}", f1)
                mlflow.log_metric(f"val_precision_{class_name}", precision)
                mlflow.log_metric(f"val_recall_{class_name}", recall)

        # Confusion matrix
        cm = confusion_matrix(y_val_encoded, y_val_pred)

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.title(f"Confusion Matrix: {run_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        mlflow.log_artifact(confusion_matrix_path)

        # Log model
        signature = infer_signature(X_train_vec, y_train_pred)
        input_example = X_train_vec[:5]
        # Log model
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        # Clean up
        if os.path.exists(confusion_matrix_path):
            os.remove(confusion_matrix_path)

        return val_metrics["f1_macro"]


def main():
    """Main training function"""

    # Set seed
    set_seed(42)

    # Load data
    train_df, val_df = load_data()

    # Define experiments
    experiments = [
        # XGBoost experiments
        {
            "model_type": "xgboost",
            "preprocessing": "none",
            "resampling": "none",
            "ngram_range": (1, 2),
        },
        {
            "model_type": "xgboost",
            "preprocessing": "none",
            "resampling": "smote",
            "ngram_range": (1, 2),
        },
        {
            "model_type": "xgboost",
            "preprocessing": "stem",
            "resampling": "none",
            "ngram_range": (1, 2),
        },
        # LightGBM experiments
        {
            "model_type": "lightgbm",
            "preprocessing": "none",
            "resampling": "none",
            "ngram_range": (1, 2),
        },
        {
            "model_type": "lightgbm",
            "preprocessing": "none",
            "resampling": "smote",
            "ngram_range": (1, 2),
        },
        # Best baseline model with SMOTE
        {
            "model_type": "logistic",
            "preprocessing": "none",
            "resampling": "smote",
            "ngram_range": (1, 2),
        },
        {
            "model_type": "logistic",
            "preprocessing": "none",
            "resampling": "random_oversample",
            "ngram_range": (1, 2),
        },
        # Random Forest with resampling
        {
            "model_type": "random_forest",
            "preprocessing": "none",
            "resampling": "smote",
            "ngram_range": (1, 2),
        },
        # Preprocessing variations with best model
        {
            "model_type": "xgboost",
            "preprocessing": "lemma",
            "resampling": "none",
            "ngram_range": (1, 2),
        },
        {
            "model_type": "xgboost",
            "preprocessing": "no_stopwords",
            "resampling": "none",
            "ngram_range": (1, 2),
        },
    ]

    # Run experiments
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'#' * 60}")
        print(f"Running Experiment {i}/{len(experiments)}")
        print(f"{'#' * 60}")

        try:
            f1_score = train_experiment(train_df, val_df, **exp_config)
            results.append(
                {"experiment": i, "config": exp_config, "f1_macro": f1_score}
            )
        except Exception as e:
            print(f"Error in experiment {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    results_sorted = sorted(results, key=lambda x: x["f1_macro"], reverse=True)

    for rank, result in enumerate(results_sorted, 1):
        config = result["config"]
        print(f"\nRank {rank}: F1={result['f1_macro']:.4f}")
        print(f"  Model: {config['model_type']}")
        print(f"  Preprocessing: {config['preprocessing']}")
        print(f"  Resampling: {config['resampling']}")

    print("\n{'=' * 60}")
    print("All advanced experiments completed!")
    print("View results at: http://localhost:5000")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
