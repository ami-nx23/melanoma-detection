
"""End-to-end EDA, model selection and held-out test evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import extract_features

RANDOM_STATE = 42
LABELS = {"benign": 0, "malignant": 1}


def discover_images(data_dir: Path, split: str):
    records = []
    for class_name, label in LABELS.items():
        folder = data_dir / split / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing expected folder: {folder}")
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"path": str(path), "label": label, "class_name": class_name, "split": split})
    return records


def load_or_extract(data_dir: Path, output_dir: Path, refresh=False):
    cache = output_dir / "features.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
        expected = sum(len(discover_images(data_dir, s)) for s in ("train", "test"))
        if len(df) == expected:
            return df
        print("Feature cache does not match current dataset; rebuilding.")
    rows = []
    for split in ("train", "test"):
        records = discover_images(data_dir, split)
        print(f"Extracting {len(records)} {split} images...")
        for i, record in enumerate(records, start=1):
            try:
                feats = extract_features(record["path"])
                rows.append({**record, **feats})
            except Exception as exc:
                print(f"Skipping {record['path']}: {exc}")
            if i % 500 == 0:
                print(f"  {i}/{len(records)}")
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def create_eda(df, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    train = df[df["split"] == "train"].copy()
    feature_cols = [c for c in train.select_dtypes(include=np.number).columns if c != "label"]

    plt.figure(figsize=(7, 4))
    sns.countplot(data=train, x="class_name", order=["benign", "malignant"])
    plt.title("Training-set class distribution")
    plt.xlabel("Image label")
    plt.ylabel("Number of images")
    savefig(figures_dir / "01_class_distribution.png")

    # Representative images from each label.
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for row, class_name in enumerate(["benign", "malignant"]):
        examples = train[train.class_name == class_name].sample(
            min(5, (train.class_name == class_name).sum()), random_state=RANDOM_STATE
        )
        for ax, (_, item) in zip(axes[row], examples.iterrows()):
            ax.imshow(Image.open(item["path"]))
            ax.set_title(class_name, fontsize=9)
            ax.axis("off")
        for ax in axes[row, len(examples):]:
            ax.axis("off")
    fig.suptitle("Example images from the training set")
    savefig(figures_dir / "02_image_examples.png")

    corr = train[feature_cols].corr(method="spearman")
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="vlag", center=0, square=False, linewidths=0.15,
                cbar_kws={"shrink": 0.7})
    plt.title("Spearman correlation between extracted image features")
    savefig(figures_dir / "03_feature_correlation.png")

    # Rank features by absolute point-biserial/Pearson correlation with label.
    associations = train[feature_cols + ["label"]].corr(numeric_only=True)["label"].drop("label")
    associations = associations.dropna().sort_values(key=np.abs, ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=associations.values, y=associations.index, orient="h")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 15 univariate feature–label associations")
    plt.xlabel("Pearson correlation with binary label")
    plt.ylabel("Feature")
    savefig(figures_dir / "04_feature_label_association.png")

    # Compare distributions of selected interpretable features.
    selected = [c for c in ["darkness_mean", "darkness_q10", "color_variance",
                             "mask_area_fraction", "mask_irregularity",
                             "edge_density", "r_mean", "g_mean", "b_mean"] if c in train.columns]
    if "color_variance" not in selected:
        selected.append("gray_std")
    for col in selected:
        if col not in train.columns:
            continue
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=train, x="class_name", y=col, order=["benign", "malignant"])
        sns.stripplot(data=train.sample(min(1000, len(train)), random_state=RANDOM_STATE),
                      x="class_name", y=col, order=["benign", "malignant"],
                      color="black", alpha=0.18, size=2)
        plt.title(f"{col}: distribution by image label")
        plt.xlabel("Image label")
        plt.ylabel(col)
        savefig(figures_dir / f"distribution_{col}.png")

    mean_features = train.groupby("class_name")[selected].mean().T
    mean_features.plot(kind="bar", figsize=(11, 5))
    plt.title("Mean extracted feature values by class")
    plt.xlabel("Feature")
    plt.ylabel("Mean value (feature scales differ)")
    plt.xticks(rotation=45, ha="right")
    savefig(figures_dir / "05_mean_features_by_class.png")


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "k-NN (k=7)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "RBF-SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=2.0, kernel="rbf", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/melanoma_cancer_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--refresh", action="store_true", help="Recompute image features.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    df = load_or_extract(args.data_dir, args.output_dir, refresh=args.refresh)
    if df.empty:
        raise RuntimeError("No images were processed. Check the dataset folder structure.")

    create_eda(df, figures_dir)
    train = df[df.split == "train"].copy()
    test = df[df.split == "test"].copy()
    feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "label"]
    X = train[feature_cols]
    y = train["label"].astype(int)
    X_test = test[feature_cols]
    y_test = test["label"].astype(int)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    validation_rows = []
    fitted = {}
    for name, pipeline in build_models().items():
        pipeline.fit(X_fit, y_fit)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_val, probabilities)
        validation_rows.append({
            "model": name,
            "validation_roc_auc": auc,
            "validation_balanced_accuracy": balanced_accuracy_score(y_val, predictions),
            "validation_f1": f1_score(y_val, predictions, zero_division=0),
        })
        fitted[name] = pipeline
        print(f"{name}: validation ROC-AUC={auc:.3f}")

    comparison = pd.DataFrame(validation_rows).sort_values("validation_roc_auc", ascending=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    print(f"Selected on validation ROC-AUC: {best_name}")

    final_model = clone(build_models()[best_name])
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "selected_model": best_name,
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "sensitivity_recall_malignant": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity_benign": float(tn / (tn + fp)) if (tn + fp) else None,
        "precision_malignant": float(precision_score(y_test, predictions, zero_division=0)),
        "f1_malignant": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels_order": ["benign", "malignant"],
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": 0.5,
        "note": "Exploratory image-level evaluation; not clinical performance."
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "test_classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["benign", "malignant"]), encoding="utf-8"
    )
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "features": feature_cols, "threshold": 0.5},
                model_dir / "selected_model.joblib")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=comparison, x="validation_roc_auc", y="model")
    plt.xlim(0, 1)
    plt.title("Model comparison on validation set")
    plt.xlabel("Validation ROC-AUC")
    plt.ylabel("Model")
    savefig(figures_dir / "06_model_comparison_validation.png")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title(f"Test confusion matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    savefig(figures_dir / "07_confusion_matrix_test.png")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("ROC curve on held-out test set")
    plt.legend()
    savefig(figures_dir / "08_roc_curve_test.png")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"Average precision = {metrics['average_precision']:.3f}")
    plt.xlabel("Recall / sensitivity")
    plt.ylabel("Precision")
    plt.title("Precision–recall curve on held-out test set")
    plt.legend()
    savefig(figures_dir / "09_precision_recall_test.png")

    # Permutation importance computed on the held-out set is descriptive only;
    # it must not be used to tune the model after looking at test results.
    perm = permutation_importance(final_model, X_test, y_test, n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False).head(15)
    importance.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance_mean", y="feature", orient="h")
    plt.title("Top permutation feature importances (test set; descriptive)")
    plt.xlabel("Decrease in ROC-AUC after permutation")
    plt.ylabel("Feature")
    savefig(figures_dir / "10_permutation_importance.png")

    print("\nFinal held-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nReports and figures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
