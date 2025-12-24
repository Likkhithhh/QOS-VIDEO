# Enhanced Server-Client Framework for Optimizing QoS (ML Version)
This project implements an ML-based QoS optimizer for video streaming inspired by the paper:
"Enhanced Server-Client Framework for Optimizing QoS in Video Streaming Over Diverse Networks".

## Contents
- utils.py                : dataset generator utilities
- data_generator.py      : generate synthetic QoS dataset
- train_models.py        : train models (classifier + regressors) and save them
- simulate_run.py        : run simulation/inference using trained models and create plots
- evaluate_models.py     : compute evaluation metrics on held-out data
- requirements.txt       : Python dependencies

## How to run (recommended using virtualenv)
1. Install dependencies:
   pip install -r requirements.txt

2. Generate dataset (optional - dataset is created by data_generator.py):
   python data_generator.py
   -> Generates `qos_training_data.csv`

3. Train models:
   python train_models.py
   -> Saves: server_classifier.pkl, regr_delay.pkl, regr_throughput.pkl, regr_delivery.pkl

4. Evaluate models:
   python evaluate_models.py

5. Run a simulation using the trained models:
   python simulate_run.py
   -> Saves simulation_results.csv and plots: plot_delay.png, plot_throughput.png, plot_delivery_ratio.png

## What to include in report/PPT
- Abstract & intro (use paper + explanation of ML approach)
- Dataset description (explain synthetic generation)
- Model selection (RandomForestClassifier + RandomForestRegressor)
- Training details and results (accuracy, RMSE, R2)
- Simulation results & plots (explain adaptive bitrate/server selection)
- Conclusion & future work (use real NS2 traces, deep learning model, combine with CSO hybrid)
