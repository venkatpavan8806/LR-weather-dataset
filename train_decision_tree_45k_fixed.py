#!/usr/bin/env python3
"""
train_decision_tree_45k_fixed.py

Full, self-contained pipeline to train and compare a pruned (max_depth=4)
and a full Decision Tree on the first 45000 rows of the provided CSV.

Saves:
 - outputs/decision_tree_pruned.png    (colored, proportions=False -> counts)
 - outputs/decision_tree_full.png      (colored, proportions=True  -> proportions)
 - outputs/decision_tree_pruned.dot
 - outputs/decision_tree_full.dot
 - outputs/decision_tree_pipeline_pruned.joblib
 - outputs/decision_tree_pipeline_full.joblib

Usage:
    python train_decision_tree_45k_fixed.py --csv /mnt/data/Loan_approval_data_2025.csv

The script is deterministic (random_state=42) and prints metrics for train/test.
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prep(csv_path: str, n_rows: int = 45000):
    df = pd.read_csv(csv_path)
    # Use first n_rows if available
    if len(df) > n_rows:
        df = df.head(n_rows).copy()

    # Drop ID column if exists
    for candidate_id in ["customer_id", "id", "loan_id"]:
        if candidate_id in df.columns:
            df.drop(columns=[candidate_id], inplace=True)
            print(f"Dropped ID column: {candidate_id}")
            break

    # Basic target detection: assume column named 'approved' or 'loan_approved' or 'label' or 'target'
    target_candidates = [c for c in df.columns if c.lower() in ("approved","loan_approved","label","target","default")] 
    if target_candidates:
        target = target_candidates[0]
    else:
        # fallback: last column
        target = df.columns[-1]
        print(f"Warning: target column not found by name heuristics; using last column '{target}' as target")

    X = df.drop(columns=[target])
    y = df[target]

    print(f"Loaded CSV from: {csv_path}")
    print(f"Selected FIRST {len(df)} rows.")

    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric columns detected: {len(numeric_cols)} -> {numeric_cols}")
    print(f"Categorical columns detected: {len(categorical_cols)} -> {categorical_cols}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # scaling is optional for trees, but harmless and can help other models if you swap later
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, *, max_depth=None, save_prefix="decision_tree"):
    # Build pipeline
    clf = DecisionTreeClassifier(random_state=42, criterion="gini", max_depth=max_depth)
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])

    print(f"\nTraining DecisionTree (max_depth={max_depth}) pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Feature importances: need feature names after preprocessing
    # extract fitted preprocessor to get feature names
    preproc = pipeline.named_steps["preproc"]
    try:
        feature_names = preproc.get_feature_names_out()
    except Exception:
        # older sklearn fallback: construct manually
        num_cols = preproc.transformers_[0][2]
        cat_transformer = preproc.transformers_[1][1].named_steps["onehot"]
        cat_cols = preproc.transformers_[1][2]
        cat_names = list(cat_transformer.get_feature_names(cat_cols))
        feature_names = list(num_cols) + cat_names

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]

    feat_df = pd.DataFrame(feat_imp, columns=["feature", "importance"]) if feat_imp else pd.DataFrame(columns=["feature","importance"]) 
    print("\nTop 20 feature importances:")
    if not feat_df.empty:
        print(feat_df.to_string(index=False))

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline model to: {model_path}")

    # Plot tree image and dot
    # Need to get the decision tree estimator and the preprocessed feature names
    tree_estimator = pipeline.named_steps["clf"]

    # To plot with human feature names, get feature names as array
    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = feature_names

    # Plot settings
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(1, 1, 1)

    # proportion argument controls whether value shows proportions or counts
    proportion_flag = (max_depth is None)  # show proportions for full tree, counts for pruned by default

    plot_tree(tree_estimator,
              feature_names=feat_names,
              class_names=[str(c) for c in np.unique(y_train)],
              filled=True,
              proportion=proportion_flag,
              impurity=True,
              rounded=True,
              fontsize=8,
              ax=ax)

    img_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.png")
    plt.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Saved tree image to: {img_path}")

    # Save dot file
    dot_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.dot")
    export_graphviz(tree_estimator, out_file=dot_path, feature_names=feat_names, class_names=[str(c) for c in np.unique(y_train)], filled=True)
    print(f"Saved dot file to: {dot_path} (use Graphviz to render .dot -> .pdf)")

    return pipeline, {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "feature_importances": feat_df
    }


def main(args):
    X, y = load_and_prep(args.csv, n_rows=args.rows)

    # Convert booleans or yes/no to numeric if necessary
    # Drop any columns that are constant
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"Dropping constant columns: {const_cols}")
        X.drop(columns=const_cols, inplace=True)

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Train/test split same as your pipeline (36000/9000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_size, random_state=42, stratify=y if args.stratify else None)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 1) Train pruned tree (max_depth=4)
    pruned_pipeline, pruned_stats = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, max_depth=4, save_prefix="decision_tree_pruned")

    # 2) Train full tree (no max_depth)
    full_pipeline, full_stats = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, max_depth=None, save_prefix="decision_tree_full")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train decision tree (pruned vs full) on first N rows of CSV and save outputs.")
    parser.add_argument("--csv", type=str, default="Loan_approval_data_2025.csv", help="Path to CSV")
    parser.add_argument("--rows", type=int, default=45000, help="How many rows from top to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train fraction (default 0.8 -> 36000/9000 on 45000)")
    parser.add_argument("--stratify", action="store_true", help="Stratify split on y (recommended if classification target is imbalanced)")
    args = parser.parse_args()

    main(args)
