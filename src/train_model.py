from sklearn.metrics import classification_report

from sklearn.svm import SVC
import pandas as pd
from prefect import flow, task
from config import DataLocation, SVC_Params
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import numpy as np


@task
def get_data(data_location: str):
    p = Path(data_location).glob("*")
    files = {file.name: pd.read_pickle(file) for file in p}
    return (
        files["X_train"],
        files["X_test"],
        files["y_train"],
        files["y_test"],
    )


@task
def train_model(
    model_params: SVC_Params, X_train: pd.DataFrame, y_train: pd.Series
):
    grid = GridSearchCV(SVC(), model_params.dict(), refit=True, verbose=3)
    grid.fit(X_train, y_train)
    return grid


@task
def predict(grid: GridSearchCV, X_test: pd.DataFrame):
    return grid.predict(X_test)


@task
def evaluate(predictions: np.ndarray, y_test: pd.Series):
    print(classification_report(y_test, predictions))


@flow(log_prints=True)
def train(
    data_location: DataLocation = DataLocation(),
    svc_params: SVC_Params = SVC_Params(),
):
    X_train, X_test, y_train, y_test = get_data(data_location.process_location)
    model = train_model(svc_params, X_train, y_train)
    predictions = predict(model, X_test)
    evaluate(predictions, y_test)


if __name__ == "__main__":
    train()
