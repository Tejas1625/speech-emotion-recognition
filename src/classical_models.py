from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from src.visualizations import plot_confusion_matrix


def run_classical_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm_fold_acc = []
    rf_fold_acc = []

    print("\n--- Starting 5-Fold Cross Validation ---")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Processing Fold {fold + 1}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ==========================================
        # MODEL 1: Pipeline for Tuned SVM
        # ==========================================
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', GridSearchCV(SVC(probability=True),
                                 {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001], 'kernel': ['rbf']}, cv=3,
                                 n_jobs=1, verbose=0))
        ])
        svm_pipeline.fit(X_train, y_train)
        svm_preds = svm_pipeline.predict(X_test)
        svm_fold_acc.append(accuracy_score(y_test, svm_preds))

        # ==========================================
        # MODEL 2: Pipeline for Random Forest
        # ==========================================
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=1))
        ])
        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)
        rf_fold_acc.append(accuracy_score(y_test, rf_preds))

    print("\n========================================")
    print("      CLASSICAL ML RESULTS SUMMARY      ")
    print("========================================")
    print(f"Tuned SVM Average Accuracy:      {np.mean(svm_fold_acc) * 100:.2f}%")
    print(f"Random Forest Average Accuracy:  {np.mean(rf_fold_acc) * 100:.2f}%")

    print("\n--- Retraining Best Model on ALL Data for Production ---")
    if np.mean(rf_fold_acc) > np.mean(svm_fold_acc):
        rf_pipeline.fit(X, y)
        plot_confusion_matrix(y_test, rf_preds, "Random Forest", "cml_confusion_matrix")
        return rf_pipeline  # Returns the pipeline (Scaler + Model)
    else:
        svm_pipeline.fit(X, y)
        plot_confusion_matrix(y_test, svm_preds, "Tuned SVM", "cml_confusion_matrix")
        return svm_pipeline  # Returns the pipeline (Scaler + Model)