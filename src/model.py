# src/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

def train_evaluate_model(X_train, X_test, y_train, y_test):
    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]}
    grid = GridSearchCV(log_reg, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train_res, y_train_res)
    best_lr = grid.best_estimator_

    y_pred_lr = best_lr.predict(X_test)
    print("=== Logistic Regression ===")
    print("Best Params:", grid.best_params_)
    print("Accuracy:", round(accuracy_score(y_test, y_pred_lr)*100,2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))

    # Random Forest (optional for better accuracy)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)
    print("\n=== Random Forest ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_rf)*100,2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
