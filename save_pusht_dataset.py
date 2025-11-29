import os
from datasets import load_from_disk
import numpy as np

def save_processed_dataset():
    # Path to the extracted dataset
    dataset_path = "pusht_expert_train"
    output_file = "pusht_data.npz"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        
        print(f"Dataset loaded. Extracting features...")
        
        # Extract features into numpy arrays
        actions = np.array(dataset['action'])
        states = np.array(dataset['state'])
        proprio = np.array(dataset['proprio'])
        episode_indices = np.array(dataset['episode_idx'])
        step_indices = np.array(dataset['step_idx'])
        
        # Note: We are NOT saving pixels here as they are paths and images are large.
        # If needed, we can save the paths or handle images separately.
        
        print(f"Saving to {output_file}...")
        np.savez_compressed(
            output_file,
            actions=actions,
            states=states,
            proprio=proprio,
            episode_indices=episode_indices,
            step_indices=step_indices
        )
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_processed_dataset()

