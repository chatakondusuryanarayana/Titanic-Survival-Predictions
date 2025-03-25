from src.preprocess import load_and_preprocess_data
from src.model import train_models

if __name__ == "__main__":
    file_path = "data/tested.csv"
    df = load_and_preprocess_data(file_path)
    train_models(df)