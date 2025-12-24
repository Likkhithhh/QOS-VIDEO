# evaluate_models.py
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv("qos_training_data.csv")

X = df[["bandwidth", "load", "latency", "packet_loss", "distance", "buffer_size", "past_qos_score"]]

clf = joblib.load("server_classifier.pkl")
regr_delay = joblib.load("regr_delay.pkl")
regr_through = joblib.load("regr_throughput.pkl")
regr_delivery = joblib.load("regr_delivery.pkl")

# ------------------------------
# SERVER CLASSIFIER ACCURACY

y_server = df["best_server"]
X_train, X_test, y_train, y_test = train_test_split(X, y_server, test_size=0.2, random_state=42)

pred_server = clf.predict(X_test)
print("Server classifier accuracy:", accuracy_score(y_test, pred_server))

# ------------------------------
# DELAY REGRESSOR
y_delay = df["delay"]
X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_delay, test_size=0.2, random_state=42)

pred_d = regr_delay.predict(X_test)
rmse_delay = math.sqrt(mean_squared_error(y_test_d, pred_d))
print("Delay RMSE:", rmse_delay, "R2:", r2_score(y_test_d, pred_d))

# ------------------------------
# THROUGHPUT REGRESSOR
y_through = df["throughput"]
X_train, X_test, y_train_t, y_test_t = train_test_split(X, y_through, test_size=0.2, random_state=42)

pred_t = regr_through.predict(X_test)
rmse_through = math.sqrt(mean_squared_error(y_test_t, pred_t))
print("Throughput RMSE:", rmse_through, "R2:", r2_score(y_test_t, pred_t))

# ------------------------------
# DELIVERY RATIO REGRESSOR

y_delivery = df["delivery_ratio"]
X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_delivery, test_size=0.2, random_state=42)

pred_r = regr_delivery.predict(X_test)
rmse_delivery = math.sqrt(mean_squared_error(y_test_r, pred_r))
print("DeliveryRatio RMSE:", rmse_delivery, "R2:", r2_score(y_test_r, pred_r))
