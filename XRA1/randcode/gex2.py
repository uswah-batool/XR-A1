import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_metrics(y_true, y_pred):
    """
    Calculates analysis metrics
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R-squared: {r2:.2f}")

    return mse, rmse, mae, r2


def plot_learning_curve(model, X, y, cv):
    """
    Plots the learning curve for a k-fold analysis
    """

    preprocessor = setup_preprocessor(X)
    preprocessor.fit(X)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean,
             label='Training score', marker='o')
    plt.plot(train_sizes, test_scores_mean,
             label='Validation score', marker='o')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def setup_preprocessor(X: pd.DataFrame):
    """ TO-DO """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_featuers = X.select_dtypes(
        include=["object", "category"]).columns

    numeric_transformer = Pipeline(steps=[('scalar', StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_featuers)
        ]
    )

    return preprocessor


def simple_linear_regression(X, y, split_type, k=None):
    """
    Handles the imple linear regression case
    """

    preprocessor = setup_preprocessor(X)

    model = Pipeline(steps=[("preprocessor", preprocessor),
                     ("regressor", LinearRegression())])

    if split_type == "train_test":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        mse, rmse, mae, r2 = calculate_metrics(y_test, y_test_pred)

    elif split_type == "train_val_test":
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        mse, rmse, mae, r2 = calculate_metrics(y_test, y_test_pred)

    elif split_type == "k_fold":
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        mse_scores = -cross_val_score(model, X, y,
                                      scoring='neg_mean_squared_error', cv=kf)
        rmse_scores = np.sqrt(mse_scores)
        average_rmse = np.mean(rmse_scores)


        mae_scores = -cross_val_score(model, X, y,
                                      scoring='neg_mean_absolute_error', cv=kf)
        average_mae = np.mean(mae_scores)


        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
        average_r2 = np.mean(r2_scores)

        plot_learning_curve(model, X, y, kf)


def multiple_linear_regression(X, y, split_type, k=None):
    """
    Handles the multiple linear regression case
    """

    preprocessor = setup_preprocessor(X)

    model = Pipeline(steps=[("preprocessor", preprocessor),
                     ("regressor", LinearRegression())])

    if split_type == "train_val_test":
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        calculate_metrics(y_test, y_test_pred)

    elif split_type == "k_fold":
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        mse_scores = -cross_val_score(model, X, y,
                                      scoring='neg_mean_squared_error', cv=kf)
        rmse_scores = np.sqrt(mse_scores)
        average_rmse = np.mean(rmse_scores)


        mae_scores = -cross_val_score(model, X, y,
                                      scoring='neg_mean_absolute_error', cv=kf)
        average_mae = np.mean(mae_scores)

        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
        average_r2 = np.mean(r2_scores)

        plot_learning_curve(model, X, y, kf)


def main():
    """
    Main function to perform linear regression analysis on a given dataset.
    Prompts the user to input the file path of the dataset, the target column number, 
    the model option (Simple Linear Regression or Multiple Linear Regression), and 
    the analysis option. Based on the user's input, it performs the specified 
    regression analysis.c
    """

    file_path = input("Enter the file path: ")
    df = pd.read_csv(file_path)

    df.dropna(inplace=True)

    for i, col in enumerate(df.columns):
        print(f"{i+1}: {col}")

    target_idx = int(input("Enter the target column number: ")) - 1
    target_column = df.columns[target_idx]
    y = df[target_column]

    model_option = int(input(
        "Enter the model option (1 - Simple Linear Regression, 0 - Multiple Linear Regression) "))

    if model_option == 1:
        predictor_idx = int(input("Enter the predictor column number: ")) - 1
        predictor_column = df.columns[predictor_idx]
        X = df[[predictor_column]]
    else:
        X = df.drop(columns=[target_column])

    analysis_options = [
        "Simple Linear Regression with Train-Test Split",
        "Simple Linear Regression with Train-Validation-Test Split",
        "Simple Linear Regression with K-Fold Cross Validation",
        "Multiple Linear Regression with Train-Validation-Test Split",
        "Multiple Linear Regression with K-Fold Cross Validation"
    ]

    for i, option in enumerate(analysis_options):
        print(f"{i+1}: {option}")
    user_option = int(input("Enter the analysis option: "))

    match user_option:
        case 1:
            simple_linear_regression(X, y, "train_test")
        case 2:
            simple_linear_regression(X, y, "train_val_test")
        case 3:
            k = int(input("Enter the number of folds: "))
            simple_linear_regression(X, y, "k_fold", k)
        case 4:
            multiple_linear_regression(X, y, "train_val_test")
        case 5:
            k = int(input("Enter the number of folds: "))
            multiple_linear_regression(X, y, "k_fold", k)
        case _:
            print("Invalid option")


if __name__ == "__main__":
    main()
