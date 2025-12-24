# video_stream_simulation.py
import cv2
import joblib
import pandas as pd
import random
import time
 
 
# Load trained ML models
clf = joblib.load("server_classifier.pkl")
regr_delay = joblib.load("regr_delay.pkl")
regr_through = joblib.load("regr_throughput.pkl")
regr_delivery = joblib.load("regr_delivery.pkl")

FEATURE_COLS = [
    "bandwidth",
    "load",
    "latency",
    "packet_loss",
    "distance",
    "buffer_size",
    "past_qos_score",
]

# Open video file
video_path = "mixkit-person-watering-a-small-plant-by-hand-33422-hd-ready.mp4"   # ✅ Make sure this file exists
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

print("✅ Video streaming started with ML-based QoS prediction...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ----------------------------
    # Simulate live network input
    # ----------------------------
    bandwidth = random.uniform(2, 20)       # Mbps
    load = random.uniform(10, 90)           # %
    latency = random.uniform(20, 200)       # ms
    packet_loss = random.uniform(0, 5)      # %
    distance = random.uniform(5, 60)        # km
    buffer_size = random.uniform(200, 1200) # KB
    past_qos_score = random.uniform(50, 100)

    row = [[bandwidth, load, latency, packet_loss, distance, buffer_size, past_qos_score]]
    X_df = pd.DataFrame(row, columns=FEATURE_COLS)

    # ----------------------------
    # ML Predictions
    # ----------------------------
    best_server = clf.predict(X_df)[0]
    pred_delay = regr_delay.predict(X_df)[0]
    pred_through = regr_through.predict(X_df)[0]
    pred_delivery = regr_delivery.predict(X_df)[0]

    # ----------------------------
    # Adaptive Streaming Logic
    # ----------------------------
    if pred_delay > 150:
        time.sleep(0.08)   # simulate buffering
        status = "BUFFERING..."
    else:
        status = "SMOOTH PLAY"

    # ----------------------------
    # Display ML Info on Video
    # ----------------------------
    text1 = f"Server: S{best_server}"
    text2 = f"Delay: {pred_delay:.1f} ms"
    text3 = f"Throughput: {pred_through:.1f} kbps"
    text4 = f"Delivery: {pred_delivery:.1f} %"
    text5 = f"Status: {status}"

    cv2.putText(frame, text1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, text2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, text3, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, text4, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, text5, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("ML-Based Video Streaming QoS", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Video streaming simulation ended.")
