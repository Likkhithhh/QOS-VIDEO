# utils.py
import numpy as np
import pandas as pd
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_single_sample():
    """
    Generate a synthetic sample representing network/server/client state
    and the resulting QoS outcomes.
    """
    # Inputs / features (simulate realistic ranges)
    bandwidth = random.uniform(0.5, 20.0)      # Mbps
    load = random.uniform(5, 95)               # server load %
    latency = random.uniform(10, 250)          # ms base latency
    packet_loss = random.uniform(0.0, 8.0)     # %
    distance = random.uniform(1, 100)          # km
    buffer_size = random.uniform(100, 2000)    # KB
    past_qos_score = random.uniform(40, 100)   # arbitrary past QoS

    # Derive QoS outcomes (labels) with simple physics-ish relationships
    # These are synthetic; in a real project you'd use measured data.
    # Add noise
    noise = lambda x, s=1.0: x + random.gauss(0, s)

    # Delay increases with latency, packet_loss, load, and distance; decreases with bandwidth
    delay = max(5.0, noise(latency * (1 + packet_loss/50.0) + load/2.0 + distance/10.0 - bandwidth*2.0, s=10.0))

    # Throughput roughly bandwidth scaled by (1 - load/100) minus effects of packet loss
    throughput = max(0.01, noise(bandwidth * (1 - load/100.0) * (1 - packet_loss/100.0) * 1000.0, s=50.0)) # kbps

    # Delivery ratio decreases with packet_loss and load
    delivery_ratio = min(100.0, max(0.0, noise(100.0 - packet_loss*2.5 - load/3.0, s=3.0)))

    # Best server decision - synthetic rule: prefer lower distance, lower load, higher bandwidth
    # Suppose we have 3 possible servers; choose using a simple scoring rule
    servers = [
        {"id": 1, "distance": random.uniform(1, 100), "load": random.uniform(5, 95), "bandwidth": random.uniform(1, 50)},
        {"id": 2, "distance": random.uniform(1, 100), "load": random.uniform(5, 95), "bandwidth": random.uniform(1, 50)},
        {"id": 3, "distance": random.uniform(1, 100), "load": random.uniform(5, 95), "bandwidth": random.uniform(1, 50)}
    ]
    # compute scores per server
    def srv_score(s):
        return (100 - s["load"]) + s["bandwidth"] - s["distance"]/2.0
    scores = [(s["id"], srv_score(s)) for s in servers]
    best_server = max(scores, key=lambda x: x[1])[0]

    sample = {
        "bandwidth": bandwidth,
        "load": load,
        "latency": latency,
        "packet_loss": packet_loss,
        "distance": distance,
        "buffer_size": buffer_size,
        "past_qos_score": past_qos_score,
        "delay": delay,
        "throughput": throughput,
        "delivery_ratio": delivery_ratio,
        "best_server": best_server
    }

    # Also include the server features for training server selection if desired (optional)
    # For simplicity we train server classifier from aggregated single-best-server label.
    return sample

def create_dataset(n_samples=5000, path="qos_training_data.csv"):
    rows = []
    for _ in range(n_samples):
        rows.append(generate_single_sample())
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path} ({len(df)} rows).")
    return df
