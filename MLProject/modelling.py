"""
modelling.py  (Kriteria 3 - dijalankan oleh MLflow Project di GitHub Actions)

Versi modelling.py dari Kriteria 2 yang disesuaikan untuk CI:
  - Tidak mengatur tracking URI sendiri. Saat dijalankan lewat `mlflow run`,
    MLflow sudah menyiapkan run aktif dan lokasi penyimpanan (folder mlruns).
  - Hyperparameter diterima dari argumen, yang diisi oleh file MLProject.
    Nilai default-nya adalah parameter terbaik dari modelling_tuning.py.
"""

import argparse
import os

import mlflow
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

TARGET = "Churn"


def main(args):
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    mlflow.sklearn.autolog(log_input_examples=True)

    # Di dalam `mlflow run`, start_run() melanjutkan run yang sudah dibuat MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        print(f"Test accuracy: {acc:.4f} | Test F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="telco_churn_preprocessing")
    main(parser.parse_args())
