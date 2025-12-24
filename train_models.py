# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

DATA_PATH = "qos_training_data.csv"

def train_and_save():
    df = pd.read_csv(DATA_PATH)

    # Features
    X = df[["bandwidth", "load", "latency", "packet_loss", "distance", "buffer_size", "past_qos_score"]]

    # -------------------------
    # 1) SERVER CLASSIFIER
    # -------------------------
    y_server = df["best_server"]

    X_train, X_test, y_train, y_test = train_test_split(X, y_server, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"[Server Classifier] Accuracy: {acc:.4f}")

    joblib.dump(clf, "server_classifier.pkl")

    # -------------------------
    # 2) DELAY REGRESSOR
    # -------------------------
    y_delay = df["delay"]

    X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_delay, test_size=0.2, random_state=42)

    regr_delay = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    regr_delay.fit(X_train, y_train_d)

    pred_d = regr_delay.predict(X_test)
    rmse_delay = mean_squared_error(y_test_d, pred_d) ** 0.5

    print("[Delay Regressor] RMSE:", rmse_delay, " R2:", r2_score(y_test_d, pred_d))

    joblib.dump(regr_delay, "regr_delay.pkl")

    # -------------------------
    # 3) THROUGHPUT REGRESSOR
    # -------------------------
    y_throughput = df["throughput"]

    X_train, X_test, y_train_t, y_test_t = train_test_split(X, y_throughput, test_size=0.2, random_state=42)

    regr_through = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    regr_through.fit(X_train, y_train_t)

    pred_t = regr_through.predict(X_test)
    rmse_through = mean_squared_error(y_test_t, pred_t) ** 0.5

    print("[Throughput Regressor] RMSE:", rmse_through, " R2:", r2_score(y_test_t, pred_t))

    joblib.dump(regr_through, "regr_throughput.pkl")

    # -------------------------
    # 4) DELIVERY RATIO REGRESSOR
    # -------------------------
    y_delivery = df["delivery_ratio"]

    X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_delivery, test_size=0.2, random_state=42)

    regr_delivery = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    regr_delivery.fit(X_train, y_train_r)

    pred_r = regr_delivery.predict(X_test)
    rmse_delivery = mean_squared_error(y_test_r, pred_r) ** 0.5

    print("[Delivery Ratio Regressor] RMSE:", rmse_delivery, " R2:", r2_score(y_test_r, pred_r))

    joblib.dump(regr_delivery, "regr_delivery.pkl")

    print("\nModels saved successfully!")

if __name__ == "__main__":
    train_and_save()
