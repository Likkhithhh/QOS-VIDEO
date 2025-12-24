# data_generator.py
from utils import create_dataset

if __name__ == "__main__":
    # create a dataset with 8000 rows for better training
    create_dataset(n_samples=8000, path="qos_training_data.csv")
