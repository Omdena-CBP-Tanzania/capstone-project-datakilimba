"""
Module for preparing features and building model pipelines.
"""
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from constants import LABEL_MAP
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def compare_multiple_regressors(df, region, target, save_best=True, model_dir="models"):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR

    df_region = df[df["Region"] == region]
    X, y = prepare_features(df_region, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(),
        "SupportVector": SVR(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        pipeline = make_model_pipeline(model)
        trained = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test, task="regression")
        trained_models[name] = trained
        results.append({
            "Model": name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "R2": metrics.get("r2", None)
        })

    results_df = pd.DataFrame(results).sort_values("RMSE")

    if save_best:
        best_model_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_model_name]

        # ✅ Store model name internally before saving
        if hasattr(best_model.named_steps.get("model", None), "__class__"):
            setattr(best_model.named_steps["model"], "_model_name", best_model_name)

        save_name = f"{region}_{target}_best.pkl"
        save_path = os.path.join(model_dir, save_name)
        save_model(best_model, save_path)
        print(f"✅ Best model ({best_model_name}) saved to {save_path}")

    return results_df


def prepare_features(df, target, drop_cols=["YearMonth", "Region", "Year"], show_correlation=True):
    X = df.drop(columns=drop_cols + [target], errors="ignore")
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)

    if show_correlation:
        print("\n🔍 Feature Correlation with Target ('{}'):".format(target))
        if y.dtype in ['float64', 'int64']:
            numeric_X = X.select_dtypes(include=["number"])
            corr = numeric_X.corrwith(y).sort_values(ascending=False)
            friendly_labels = [LABEL_MAP.get(col, col) for col in corr.index]
            print(pd.Series(corr.values, index=friendly_labels))
            plt.figure(figsize=(10, 4))
            sns.barplot(x=corr.values, y=friendly_labels)
            plt.title(f"Feature Correlation with {LABEL_MAP.get(target, target)}")
            plt.xlabel("Correlation Coefficient")
            plt.tight_layout()
            plt.show()
        else:
            print("(Correlation analysis skipped — target is categorical)")
    return X, y

def make_model_pipeline(model):
    scale_sensitive_models = (
        "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
        "SVC", "KNeighborsClassifier", "KNeighborsRegressor",
        "MLPClassifier", "MLPRegressor"
    )
    model_name = model.__class__.__name__
    if model_name in scale_sensitive_models:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        return Pipeline([
            ('model', model)
        ])

def split_data(X, y, test_size=0.2, random_state=42):
    stratify = y if y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() <= 10 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def train_model(model_pipeline, X_train, y_train):
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def evaluate_model(model_pipeline, X_test, y_test, task='classification'):
    y_pred = model_pipeline.predict(X_test)
    if task == 'classification':
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
    else:
        return {
    "rmse": mean_squared_error(y_test, y_pred, squared=False),
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

def save_model(model_pipeline, filepath, model_type=None):
    """
    Saves the model pipeline to the specified filepath.
    Optionally embeds the model type into the pipeline object for later retrieval.
    """
    if model_type:
        setattr(model_pipeline, "model_type", model_type)  # Embed model type as metadata
    joblib.dump(model_pipeline, filepath)

def load_model(filepath):
    """
    Load model and return. Also ensure model_type exists.
    """
    model = joblib.load(filepath)
    if not hasattr(model, "model_type"):
        model.model_type = "Unknown"
    return model

def tune_model(model, param_grid, X_train, y_train, scoring='neg_root_mean_squared_error', method='random', n_iter=20, cv=5):
    """
    Generic hyperparameter tuning with either GridSearchCV or RandomizedSearchCV.

    Args:
        model: sklearn estimator
        param_grid: dict
        X_train: pd.DataFrame
        y_train: pd.Series
        scoring: str
        method: 'random' or 'grid'
        n_iter: only used for randomized search
        cv: int

    Returns:
        best_model, best_params, best_score
    """
    pipeline = make_model_pipeline(model)
    if method == "grid":
        search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv)
    else:
        search = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring=scoring, cv=cv, n_iter=n_iter, random_state=42)

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, -search.best_score_

def run_regression_for_region(
    df, region, target, model,
    use_search=False, param_grid=None,
    search_method='random', n_iter=10
):
    """
    Train a regression model for a specific region and target variable,
    optionally using hyperparameter search (grid or randomized).
    """
    df_region = df[df["Region"] == region]
    X, y = prepare_features(df_region, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if use_search and param_grid:
        best_model, best_params, best_score = tune_model(
            model,
            param_grid,
            X_train,
            y_train,
            method=search_method,
            n_iter=n_iter
        )
        metrics = evaluate_model(best_model, X_test, y_test, task="regression")
        metrics.update({"best_params": best_params, "cv_rmse": best_score})
        model_step = best_model.named_steps.get("model")
    else:
        pipeline = make_model_pipeline(model)
        best_model = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(best_model, X_test, y_test, task="regression")
        model_step = best_model.named_steps.get("model")

    # Display feature importances
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        top_features = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])[:5]
        print("\n🧠 Top 5 Important Features (User-Friendly Labels):")
        for name, score in top_features:
            label = LABEL_MAP.get(name, name)
            print(f"  {label}: {score:.3f}")

    return best_model, metrics

def run_classification_for_region(df, region, target, model, use_search=False, param_grid=None, search_method='random'):
    df_region = df[df["Region"] == region]
    X, y = prepare_features(df_region, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if use_search and param_grid:
        best_model, best_params, best_score = tune_model(
            model, param_grid, X_train, y_train, scoring='accuracy', method=search_method
        )
        metrics = evaluate_model(best_model, X_test, y_test, task="classification")
        metrics.update({"best_params": best_params, "cv_accuracy": best_score})
        return best_model, metrics
    else:
        pipeline = make_model_pipeline(model)
        trained = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test, task="classification")
        return trained, metrics

def compare_models(df, region, target, models_dict):
    X, y = prepare_features(df[df["Region"] == region], target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = {}
    for name, model in models_dict.items():
        pipe = make_model_pipeline(model)
        trained = train_model(pipe, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test, task="regression")
        results[name] = metrics
    return results
