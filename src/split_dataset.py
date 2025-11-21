import os
import shutil
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data/raw/food41")
IMG_DIR = os.path.join(RAW_DIR, "images")
META_DIR = os.path.join(RAW_DIR, "meta/meta")

PROCESSED_DIR = os.path.join(ROOT, "data/processed")
TRAIN_OUT = os.path.join(PROCESSED_DIR, "train")
VAL_OUT = os.path.join(PROCESSED_DIR, "val")
TEST_OUT = os.path.join(PROCESSED_DIR, "test")

def load_split_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
    
def copy_files(file_list, dest_root):
    for item in file_list:
        class_name, image_name = item.split("/")
        src = os.path.join(IMG_DIR, class_name, image_name +".jpg")
        dest = os.path.join(dest_root, class_name)

        os.makedirs(dest, exist_ok=True)
        shutil.copy2(src, dest)

def main(val_ratio=.1):
    train_files = load_split_file(os.path.join(META_DIR, "train.txt"))
    test_files = load_split_file(os.path.join(META_DIR, "test.txt"))

    train_files, val_files = train_test_split(train_files, test_size =val_ratio, random_state=42, shuffle = True)

    for folder in [TRAIN_OUT, VAL_OUT, TEST_OUT]:
        os.makedirs(folder, exist_ok=True)
    
    print("Copying TRAIN images...")
    copy_files(train_files, TRAIN_OUT)

    print("Copying VAL images...")
    copy_files(val_files, VAL_OUT)

    print("Copying TEST images...")
    copy_files(test_files, TEST_OUT)

    print("Done! Processed dataset saved to:", PROCESSED_DIR)
    
if __name__ == "__main__":
    main()