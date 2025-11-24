Create a venv

python -m venv venv/

Run venv

venv/Scripts/activate

Install all packages

pip install -r requirements.txt

Download & setup dataset

python src/setup_dataset.py

This will download food41 into data/raw/food41/

run python src/split_dataset.py

This will split raw data into train, test, and validation sets

run python src/data_loader.py

This will test the data loader and output an image visualization sample
