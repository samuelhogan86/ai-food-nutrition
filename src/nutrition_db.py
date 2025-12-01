import os
import json
import sys
import pandas as pd

# Add project root to Python path so imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.data_loader import get_data_loaders

OUTPUT_PATH = os.path.join(ROOT, "data", "nutrition", "food_nutrition.json")
NUTRITION_CSV_PATH = os.path.join(ROOT, "data", "nutrition", "nutrition.csv")

def normalize_name(name: str):
    """
    Normalize a food name so it matches Food-101 class naming style.
        "Apple Pie"  → "apple_pie"
    """
    return (
        name.lower()
            .replace("-", " ")
            .replace("_", " ")
            .strip()
            .replace(" ", "_")
    )

def load_class_names():
    # Load class names using the existing data_loader.
    processed_dir = os.path.join(ROOT, "data", "processed")

    # Only need class names, so minimal loader call
    _, _, _, class_names = get_data_loaders(
        data_dir=processed_dir,
        batch_size=1,
        num_workers=0
    )
    return class_names


def load_nutrition_csv():
    # Loads the Kaggle nutrition CSV into a pandas DataFrame.
    if not os.path.exists(NUTRITION_CSV_PATH):
        raise FileNotFoundError(f"Nutrition file not found: {NUTRITION_CSV_PATH}")

    df = pd.read_csv(NUTRITION_CSV_PATH)

    # Normalize food names in dataset
    df["normalized_name"] = df["label"].apply(normalize_name)

    return df


def match_food_to_nutrition(class_names, df):
    """
    Match each Food-101 class to its nutrition entry.

    If an item is missing in the dataset, inserts: None
    """
    nutrition_map = {}

    for cls in class_names:
        match = df[df["normalized_name"] == cls]

        if match.empty:
            nutrition_map[cls] = None
        else:
            # Convert row to dictionary excluding the normalized_name helper column
            row = match.iloc[0].drop(labels=["normalized_name"]).to_dict()
            nutrition_map[cls] = row

    return nutrition_map


def save_json(data):
    # Write the final JSON mapping.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\nNutrition database saved to:\n{OUTPUT_PATH}\n")


def build_nutrition_database():
    print("Loading class names...")
    class_names = load_class_names()

    print(f"Loaded {len(class_names)} classes.")

    print("Loading nutrition CSV...")
    df = load_nutrition_csv()

    print("Matching classes to nutrition entries...")
    nutrition_map = match_food_to_nutrition(class_names, df)

    print("Saving JSON...")
    save_json(nutrition_map)

    print("Done!")


if __name__ == "__main__":
    build_nutrition_database()