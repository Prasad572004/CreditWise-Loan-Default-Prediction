# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    df[['Age', 'Income', 'CreditScore', 'LoanAmount', 'Loan_Term']] = imputer.fit_transform(
        df[['Age', 'Income', 'CreditScore', 'LoanAmount', 'Loan_Term']]
    )

    # Feature engineering
    df["DebtToIncome"] = df["LoanAmount"] / (df["Income"] + 1)
    df["LoanToAge"] = df["LoanAmount"] / (df["Age"] + 1)
    df["IncomePerTerm"] = df["Income"] / (df["Loan_Term"] + 1)
    
    # Optional: bucket credit score
    df["CreditScore_Bucket"] = pd.cut(df["CreditScore"], bins=[300, 600, 700, 850],
                                      labels=["Poor", "Fair", "Good"])
    df = pd.get_dummies(df, columns=["CreditScore_Bucket"], drop_first=True)

    X = df.drop("Default", axis=1)
    y = df["Default"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
