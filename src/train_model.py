import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from config import DataLocation, SVC_Params


@task
def get_processed_data(data_location: str):
    return {
        file: pd.read_pickle(f"{data_location}/{file}")
        for file in ["X_train", "X_test", "y_train", "y_test"]
    }


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
    data = get_processed_data(data_location.process_location)
    model = train_model(svc_params, data["X_train"], data["y_train"])
    predictions = predict(model, data["X_test"])
    evaluate(predictions, data["y_test"])


if __name__ == "__main__":
    train()
