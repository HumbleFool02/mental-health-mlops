"""
Baseline Model Training Script
Experiments with traditional ML algorithms for text classification
"""

import json
import os
import sys
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.utils import set_seed

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


def load_class_weights():
    """Load pre-calculated class weights"""
    with open("configs/class_weights.json", "r") as f:
        class_weights = json.load(f)
    return class_weights


def create_vectorizer(vectorizer_type="tfidf", max_features=5000, ngram_range=(1, 1)):
    """Create text vectorizer"""
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95
        )
    elif vectorizer_type == "count":
        return CountVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def create_model(model_type="logistic", class_weights=None):
    """Create ML model"""
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "naive_bayes":
        return MultinomialNB(alpha=1.0)
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "svm":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
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
    vectorizer_type="tfidf",
    model_type="logistic",
    ngram_range=(1, 1),
    max_features=5000,
    experiment_name="baseline-experiments",
):
    """
    Run a single training experiment and log to MLflow
    """

    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{vectorizer_type}_{model_type}"):
        print(f"\n{'=' * 60}")
        print(f"Experiment: {vectorizer_type} + {model_type}")
        print(f"{'=' * 60}")
        start_time = time.time()

        # Log parameters
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))

        # Prepare data
        X_train = train_df["text"].values
        y_train = train_df["label"].values
        X_val = val_df["text"].values
        y_val = val_df["label"].values

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        # Create and fit vectorizer
        print("Vectorizing text...")
        vectorizer = create_vectorizer(vectorizer_type, max_features, ngram_range)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        print(f"Feature matrix shape: {X_train_vec.shape}")
        mlflow.log_param("actual_features", X_train_vec.shape[1])

        # Create and train model
        print("Training model...")
        model = create_model(model_type)
        model.fit(X_train_vec, y_train_encoded)

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # Predictions
        print("Making predictions...")
        y_train_pred = model.predict(X_train_vec)
        y_val_pred = model.predict(X_val_vec)

        # Evaluate on training set
        train_metrics, train_report = evaluate_model(
            y_train_encoded, y_train_pred, label_encoder
        )
        print("\nTraining Metrics:")
        print("  Accuracy: {train_metrics['accuracy']:.4f}")
        print("  F1 (macro): {train_metrics['f1_macro']:.4f}")

        # Evaluate on validation set
        val_metrics, val_report = evaluate_model(
            y_val_encoded, y_val_pred, label_encoder
        )
        print("\nValidation Metrics:")
        print("  Accuracy: {val_metrics['accuracy']:.4f}")
        print("  F1 (macro): {val_metrics['f1_macro']:.4f}")
        print("  F1 (weighted): {val_metrics['f1_weighted']:.4f}")

        # Log metrics to MLflow
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        # Log per-class metrics for validation
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

        # Save confusion matrix as artifact
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
        plt.title(f"Confusion Matrix: {vectorizer_type} + {model_type}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        mlflow.log_artifact(confusion_matrix_path)

        signature = infer_signature(X_train_vec, y_train_pred)
        input_example = X_train_vec[:5]
        # Log model
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )

        # Log vectorizer (as artifact)
        import joblib

        vectorizer_path = "vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)

        # Log label encoder
        encoder_path = "label_encoder.pkl"
        joblib.dump(label_encoder, encoder_path)
        mlflow.log_artifact(encoder_path)

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        # Clean up temporary files
        import os

        for temp_file in [confusion_matrix_path, vectorizer_path, encoder_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return val_metrics["f1_macro"]


def main():
    """Main training function"""

    # Set seed for reproducibility
    set_seed(42)

    # Load data
    train_df, val_df = load_data()

    # Load class weights
    class_weights = load_class_weights()
    print("\nClass weights loaded:")
    for class_name, weight in sorted(
        class_weights.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {class_name:20s}: {weight:.3f}")

    # Define experiments to run
    experiments = [
        # Experiment 1: TF-IDF + Logistic Regression (baseline)
        {
            "vectorizer_type": "tfidf",
            "model_type": "logistic",
            "ngram_range": (1, 1),
            "max_features": 5000,
        },
        # Experiment 2: TF-IDF with bigrams + Logistic Regression
        {
            "vectorizer_type": "tfidf",
            "model_type": "logistic",
            "ngram_range": (1, 2),
            "max_features": 5000,
        },
        # Experiment 3: TF-IDF + Naive Bayes
        {
            "vectorizer_type": "tfidf",
            "model_type": "naive_bayes",
            "ngram_range": (1, 1),
            "max_features": 5000,
        },
        # Experiment 4: TF-IDF + Random Forest
        {
            "vectorizer_type": "tfidf",
            "model_type": "random_forest",
            "ngram_range": (1, 1),
            "max_features": 5000,
        },
        # Experiment 5: TF-IDF + SVM
        {
            "vectorizer_type": "tfidf",
            "model_type": "svm",
            "ngram_range": (1, 1),
            "max_features": 5000,
        },
        # Experiment 6: Count Vectorizer + Logistic Regression
        {
            "vectorizer_type": "count",
            "model_type": "logistic",
            "ngram_range": (1, 1),
            "max_features": 5000,
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
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    results_sorted = sorted(results, key=lambda x: x["f1_macro"], reverse=True)

    for rank, result in enumerate(results_sorted, 1):
        config = result["config"]
        print(f"\nRank {rank}: F1={result['f1_macro']:.4f}")
        print(f"  Vectorizer: {config['vectorizer_type']}")
        print(f"  Model: {config['model_type']}")
        print(f"  N-grams: {config['ngram_range']}")

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print("View results at: http://localhost:5000")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
