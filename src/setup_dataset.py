import kagglehub
import os
import shutil

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../data/raw/food41")

def setup_dataset():
    print("Downloading dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download("kmader/food41")
    print("Download complete.")
    print("Downloaded to:", downloaded_path)

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("\ncopying dataset into proj dir")
    shutil.copytree(downloaded_path, DATASET_DIR, dirs_exist_ok=True)

    print(f"Dataset is now available at: {DATASET_DIR}")

if __name__ == "__main__":
    setup_dataset()