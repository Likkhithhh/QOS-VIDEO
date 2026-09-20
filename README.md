# ML-Based QoS Optimizer for Video Streaming

A machine-learning simulation for studying **Quality of Service (QoS)** optimization in video-streaming environments.

The project models a workflow in which network/QoS observations are generated, ML models are trained, performance is evaluated, and a simulation produces delivery, delay, and throughput results.

## Pipeline

```text
Synthetic QoS data
      ↓
Feature preparation
      ↓
Classifier + regression models
      ↓
Evaluation
      ↓
Streaming simulation
      ↓
QoS metrics and plots
```

## Main components

- `data_generator.py` — generates synthetic QoS training data
- `train_models.py` — trains the classifier and regression models
- `evaluate_models.py` — evaluates trained model performance
- `simulate_run.py` — runs the QoS simulation
- `video_stream_simulation.py` — video-streaming simulation logic
- `utils.py` — shared utilities
- `requirements.txt` — Python dependencies

## Tech stack

Python, NumPy, pandas, scikit-learn, Matplotlib, seaborn, joblib, and tqdm.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Run the project

Generate or refresh the dataset:

```bash
python3 data_generator.py
```

Train models:

```bash
python3 train_models.py
```

Evaluate:

```bash
python3 evaluate_models.py
```

Run a simulation:

```bash
python3 simulate_run.py
```

## Generated artifacts

Training and simulation create model binaries, result CSV files, and plots. These are build/runtime artifacts and are excluded from future commits through `.gitignore`.

## Scope and limitations

This is an educational/simulation project. The included workflow uses synthetic QoS data and should not be interpreted as validation on production network traffic. A natural next step is evaluation with real network traces and controlled streaming experiments.
