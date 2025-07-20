import pandas as pd
import json
import os

def inspect_data(dataset_path, sample_id):
    """
    Inspects the parquet and json files for a given sample in the dataset.
    """
    sample_path = os.path.join(dataset_path, sample_id)
    print(f"--- Inspecting data for sample: {sample_id} ---")

    # Inspect parquet file
    parquet_files = [f for f in os.listdir(sample_path) if f.endswith('.parquet')]
    if parquet_files:
        parquet_path = os.path.join(sample_path, parquet_files[0])
        print(f"Reading parquet file: {parquet_path}")
        try:
            df = pd.read_parquet(parquet_path)
            print("Parquet file shape:")
            print(df.shape)
            print("\nParquet file info:")
            df.info()
        except Exception as e:
            print(f"Error reading parquet file: {e}")
    else:
        print("No parquet file found.")

    # Inspect json file
    json_files = [f for f in os.listdir(sample_path) if f.endswith('.json')]
    if json_files:
        json_path = os.path.join(sample_path, json_files[0])
        print(f"\nReading json file: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print("JSON file keys:")
                print(data.keys())
        except Exception as e:
            print(f"Error reading json file: {e}")
    else:
        print("No json file found.")

if __name__ == "__main__":
    # Paths to the datasets
    train_path = os.path.join("dataset", "train_small")
    val_path = os.path.join("dataset", "val_small")
    test_path = os.path.join("dataset", "test_small")

    # Get one sample from each dataset
    train_sample = os.listdir(train_path)[0]
    val_sample = os.listdir(val_path)[0]
    test_sample = os.listdir(test_path)[0]

    # Inspect the data
    inspect_data(train_path, train_sample)
    inspect_data(val_path, val_sample)
    inspect_data(test_path, test_sample)