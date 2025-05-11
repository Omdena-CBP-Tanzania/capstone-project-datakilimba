# model_utils.py
"""
Module for preparing features and building model pipelines.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from constants import LABEL_MAP

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
            "mae": mean_absolute_error(y_test, y_pred)
        }

def save_model(model_pipeline, filepath):
    joblib.dump(model_pipeline, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def tune_model_with_gridsearch(model, param_grid, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5):
    pipeline = make_model_pipeline(model)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, -grid.best_score_

def run_regression_for_region(df, region, target, model, use_gridsearch=False, param_grid=None):
    df_region = df[df["Region"] == region]
    X, y = prepare_features(df_region, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if use_gridsearch and param_grid:
        best_model, best_params, best_score = tune_model_with_gridsearch(
            model, param_grid, X_train, y_train
        )
        metrics = evaluate_model(best_model, X_test, y_test, task="regression")
        metrics.update({"best_params": best_params, "cv_rmse": best_score})
        model_step = best_model.named_steps.get("model")
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            top_features = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])[:5]
            print("\n🔍 Top 5 Important Features:")
            for name, score in top_features:
                label = LABEL_MAP.get(name, name)
                print(f"  {label}: {score:.3f}")
        return best_model, metrics
    else:
        pipeline = make_model_pipeline(model)
        trained = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test, task="regression")
        model_step = trained.named_steps.get("model")
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            top_features = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])[:5]
            print("\n🔍 Top 5 Important Features:")
            for name, score in top_features:
                label = LABEL_MAP.get(name, name)
                print(f"  {label}: {score:.3f}")
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

def run_classification_for_region(df, region, target, model, use_gridsearch=False, param_grid=None):
    df_region = df[df["Region"] == region]
    X, y = prepare_features(df_region, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if use_gridsearch and param_grid:
        best_model, best_params, best_score = tune_model_with_gridsearch(
            model, param_grid, X_train, y_train, scoring="accuracy"
        )
        metrics = evaluate_model(best_model, X_test, y_test, task="classification")
        metrics.update({"best_params": best_params, "cv_accuracy": best_score})
        return best_model, metrics
    else:
        pipeline = make_model_pipeline(model)
        trained = train_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test, task="classification")
        return trained, metrics
