
import joblib
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

sns.set(style="darkgrid")

# Load models
clf = joblib.load("server_classifier.pkl")
regr_delay = joblib.load("regr_delay.pkl")
regr_through = joblib.load("regr_throughput.pkl")
regr_delivery = joblib.load("regr_delivery.pkl")

# Feature names (same as training)
FEATURE_COLS = [
    "bandwidth",
    "load",
    "latency",
    "packet_loss",
    "distance",
    "buffer_size",
    "past_qos_score",
]

def generate_runtime_input():
    bandwidth = random.uniform(0.5, 20.0)      # Mbps
    load = random.uniform(5, 95)               # %
    latency = random.uniform(10, 250)          # ms
    packet_loss = random.uniform(0.0, 8.0)     # %
    distance = random.uniform(1, 100)          # km
    buffer_size = random.uniform(100, 2000)    # KB
    past_qos_score = random.uniform(40, 100)   # previous QoS indicator
    return [bandwidth, load, latency, packet_loss, distance, buffer_size, past_qos_score]

def simulate_streaming(num_steps=150):  # lowered from 400 to 150
    metrics = {
        "step": [],
        "selected_server": [],
        "pred_delay": [],
        "pred_throughput": [],
        "pred_delivery": [],
        "bandwidth": [],
        "load": [],
    }

    for i in trange(num_steps):
        x = generate_runtime_input()

        # Convert to DataFrame with proper column names to avoid warnings
        X_df = pd.DataFrame([x], columns=FEATURE_COLS)

        # ML predictions
        selected_server = clf.predict(X_df)[0]
        pred_delay = regr_delay.predict(X_df)[0]
        pred_through = regr_through.predict(X_df)[0]
        pred_delivery = regr_delivery.predict(X_df)[0]
        
        # Simple adaptive bitrate logic 
        bandwidth_mbps = x[0]
        requested_bitrate = bandwidth_mbps * 1000.0  # kbps

        if pred_delay > 150:
            safety_factor = 0.75
        else:
            safety_factor = 0.9

        used_bitrate = min(requested_bitrate, pred_through * safety_factor)

        # store metrics
        metrics["step"].append(i)
        metrics["selected_server"].append(selected_server)
        metrics["pred_delay"].append(pred_delay)
        metrics["pred_throughput"].append(pred_through)
        metrics["pred_delivery"].append(pred_delivery)
        metrics["bandwidth"].append(bandwidth_mbps)
        metrics["load"].append(x[1])

    df = pd.DataFrame(metrics)
    return df

if __name__ == "__main__":
    df = simulate_streaming(num_steps=150)
    df.to_csv("simulation_results.csv", index=False)
    print("Simulation complete. Results saved to simulation_results.csv")

    # Plot Delay
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["pred_delay"], label="Pred Delay (ms)")
    plt.axhline(150, linestyle="--", label="Delay Threshold (150 ms)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_delay.png", dpi=200)
    plt.close()

    # Plot Throughput
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["pred_throughput"], label="Pred Throughput (kbps)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Throughput (kbps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_throughput.png", dpi=200)
    plt.close()

    # Plot Delivery Ratio
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["pred_delivery"], label="Pred Delivery Ratio (%)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Delivery Ratio (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_delivery_ratio.png", dpi=200)
    plt.close()

    print("Plots saved: plot_delay.png, plot_throughput.png, plot_delivery_ratio.png")
