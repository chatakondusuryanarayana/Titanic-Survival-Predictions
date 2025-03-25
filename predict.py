import pandas as pd
import pickle
from src.preprocess import load_and_preprocess_data

def predict_survival(file_path, model_path="models/saved_model.pkl"):
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_and_preprocess_data(file_path)
    X = df.drop(columns=["Survived"])  # Remove target variable

    predictions = model.predict(X)
    df["Predicted_Survival"] = predictions

    print(df[["Predicted_Survival"]].head())

    return df

if __name__ == "__main__":
    predict_survival("data/tested.csv")