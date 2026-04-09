import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DemoClassifier:
    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path
        self.model = None

        if os.path.exists(model_path):
            self.load()

        self.sex_map = {"M": 0, "V": 1}
        self.bin_map = {"+": 1, "-": 0}

    def preprocess(self, df):
        df = df.replace("?", np.nan)
        df = df.fillna(df.mean(numeric_only=True))

        df["geslacht"] = df["geslacht"].map(self.sex_map)
        df["hypertensie"] = df["hypertensie"].map(self.bin_map)
        df["hartinfarct"] = df["hartinfarct"].map(self.bin_map)
        df["diabetes"] = df["diabetes"].map(self.bin_map)
        df["nierziekte"] = df["nierziekte"].map(self.bin_map)

        df = df.drop(columns=["Individu-ID"], errors="ignore")
        return df

    # Trainen
    def train(self, filename="data-studenten.csv"):
        df = pd.read_csv(filename)

        df["prognose10jaar"] = df["prognose10jaar"].map({"CHD+": 1, "CHD-": 0})
        df = self.preprocess(df)

        train_df, test_df = train_test_split(
            df,
            test_size=0.10,
            random_state=42,
            stratify=df["prognose10jaar"])

        X_train = train_df.drop(columns=["prognose10jaar"])
        y_train = train_df["prognose10jaar"]

        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]}

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_grid,
            cv=5,
            n_jobs=-1)

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_

        print("\nBeste parameters:", grid.best_params_)

        self.save()

        print("Model getraind en opgeslagen als model.pkl")

        # Automatisch testen
        self.test(test_df)

    # Testen op de 10% split
    def test(self, test_df):
        X_test = test_df.drop(columns=["prognose10jaar"])
        y_test = test_df["prognose10jaar"]

        y_pred = self.model.predict(X_test)

        print("\n=== TESTRESULTATEN (10%) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification report:\n", classification_report(y_test, y_pred))
        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Predict op csv bestand
    def predict(self, filename):
        df = pd.read_csv(filename)
        X = self.preprocess(df)
        preds = self.model.predict(X)
        return preds.astype(bool)

    def save(self):
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.model = joblib.load(self.model_path)

# pythonmodel.py om model te laten trainen en correlatie matrix te maken
# python3 -m WintonOverwat predict competitie.csv

if __name__ == "__main__":
    model = DemoClassifier()
    model.train("data-studenten.csv")