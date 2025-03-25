from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train_models(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize models
    log_reg = LogisticRegression()
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train models
    log_reg.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # Evaluate models
    def evaluate_model(y_true, y_pred, model_name):
        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred)
        }

    y_pred_log = log_reg.predict(X_test)
    y_pred_rf = rf_clf.predict(X_test)

    log_reg_results = evaluate_model(y_test, y_pred_log, "Logistic Regression")
    rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest")

    print("Logistic Regression:", log_reg_results)
    print("Random Forest:", rf_results)

    # Save the best model
    best_model = rf_clf if rf_results["Accuracy"] > log_reg_results["Accuracy"] else log_reg
    with open("models/saved_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model