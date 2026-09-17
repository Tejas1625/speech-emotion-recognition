import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.splitting import actor_disjoint_split
from src.visualizations import plot_confusion_matrix


def run_classical_models(X_clean, X_augmented, y, actor_ids):
    """
    Compare SVM and Random Forest fairly.

    Entire actors are held out for testing. Augmented features are used
    only for training, so an original recording and its augmentation
    cannot appear in separate train/test sets.
    """
    train_idx, _, test_idx = actor_disjoint_split(actor_ids)

    # Add augmentation only to the training set.
    X_train = np.vstack((X_clean[train_idx], X_augmented[train_idx]))
    y_train = np.concatenate((y[train_idx], y[train_idx]))

    # Each clean/augmented pair has the same actor group.
    train_groups = np.concatenate(
        (actor_ids[train_idx], actor_ids[train_idx])
    )

    # Final evaluation uses only clean, unseen speakers.
    X_test = X_clean[test_idx]
    y_test = y[test_idx]

    # Hyperparameter tuning must also keep actors separate.
    inner_cv = GroupKFold(
        n_splits=min(4, len(np.unique(train_groups)))
    )

    svm_search = GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True)),
        ]),
        param_grid={
            "svm__C": [1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
        },
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
    )

    rf_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [200],
            "max_depth": [None, 15],
        },
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
    )

    candidates = {
        "Tuned SVM": svm_search,
        "Random Forest": rf_search,
    }

    results = {}

    for model_name, search in candidates.items():
        # `groups` ensures an actor does not cross CV folds.
        search.fit(X_train, y_train, groups=train_groups)

        predictions = search.predict(X_test)

        results[model_name] = {
            "model": search.best_estimator_,
            # Use CV score to select the production model.
            "cv_macro_f1": float(search.best_score_),
            # Test metrics are reported only after selection/tuning.
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_macro_f1": float(
                f1_score(y_test, predictions, average="macro")
            ),
            "predictions": predictions,
        }

        print(f"\n{model_name}")
        print(f"Best CV macro-F1: {results[model_name]['cv_macro_f1']:.3f}")
        print(f"Actor-held-out accuracy: {results[model_name]['test_accuracy']:.3f}")
        print(f"Actor-held-out macro-F1: {results[model_name]['test_macro_f1']:.3f}")

    # Select model based on group-aware CV, not final test accuracy.
    best_name = max(
        results,
        key=lambda name: results[name]["cv_macro_f1"],
    )
    best = results[best_name]

    plot_confusion_matrix(
        y_test,
        best["predictions"],
        best_name,
        "cml_confusion_matrix",
    )

    return best["model"], {
        "selected_model": best_name,
        "accuracy": best["test_accuracy"],
        "macro_f1": best["test_macro_f1"],
        "cv_macro_f1": best["cv_macro_f1"],
    }