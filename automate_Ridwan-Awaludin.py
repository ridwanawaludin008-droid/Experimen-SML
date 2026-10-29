"""Export raw and preprocessed breast cancer data for the experiment repository."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename("target")
    raw_data = features.copy()
    raw_data["target"] = target
    raw_data["diagnosis"] = target.map(dict(enumerate(dataset.target_names)))
    raw_path = RAW_DIR / "breast_cancer_raw.csv"
    raw_data.to_csv(raw_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    for split, values, labels in (
        ("train", X_train_scaled, y_train),
        ("test", X_test_scaled, y_test),
    ):
        processed = pd.DataFrame(values, columns=dataset.feature_names)
        processed["target"] = labels.to_numpy()
        processed["diagnosis"] = processed["target"].map(
            dict(enumerate(dataset.target_names))
        )
        output_path = PROCESSED_DIR / f"breast_cancer_{split}_preprocessed.csv"
        processed.to_csv(output_path, index=False)

    print(f"Raw dataset: {raw_path} ({len(raw_data)} rows)")
    print(
        "Preprocessed datasets: "
        f"{len(X_train_scaled)} train rows, {len(X_test_scaled)} test rows"
    )
    print("Imputer and scaler were fit on training data only.")


if __name__ == "__main__":
    main()
