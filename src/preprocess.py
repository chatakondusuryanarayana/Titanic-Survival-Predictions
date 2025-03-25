import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Encode categorical variables
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # Male=1, Female=0
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Normalize numerical features
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

    return df