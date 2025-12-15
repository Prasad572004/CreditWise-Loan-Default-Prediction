# main.py
from src.preprocessing import preprocess_data
from src.model import train_evaluate_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("data/credit_data.csv")
    train_evaluate_model(X_train, X_test, y_train, y_test)
