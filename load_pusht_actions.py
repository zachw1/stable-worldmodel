import os
from datasets import load_from_disk
import numpy as np

def load_features():
    # Path to the extracted dataset
    dataset_path = "pusht_expert_train"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        
        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
        print(f"Features keys: {list(dataset.features.keys())}\n")
        
        # List of features to inspect
        feature_names = list(dataset.features.keys())
        
        loaded_features = {}
        
        for name in feature_names:
            # 'pixels' is typically images stored as paths or bytes, handle separately to avoid huge print/load
            if name == 'pixels':
                 print(f"Feature: '{name}'")
                 print(f"  Type: {dataset.features[name]}")
                 # Just check the first element to see what it looks like
                 sample = dataset[0][name]
                 print(f"  First sample: {sample}")
                 print("-" * 30)
                 continue

            print(f"Feature: '{name}'")
            data = dataset[name]
            
            # Convert to numpy for shape inspection if it's numeric
            try:
                data_np = np.array(data)
                print(f"  Shape: {data_np.shape}")
                print(f"  Dtype: {data_np.dtype}")
                
                # Print a sample
                if len(data_np) > 0:
                    print(f"  First sample: {data_np[0]}")
                
                loaded_features[name] = data_np
            except Exception as e:
                print(f"  Could not convert to numpy array: {e}")
                # Fallback for non-numeric or complex types
                print(f"  First sample: {data[0]}")

            print("-" * 30)
            
        return loaded_features, dataset

    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_features()
