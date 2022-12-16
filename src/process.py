import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split

from config import DataLocation, ProcessConfig


@task
def get_data(data_location: str, file_name: str):
    return pd.read_csv(f"{data_location}/{file_name}")


@task
def drop_columns(data: pd.DataFrame, columns: list):
    return data.drop(columns=columns)


@task
def get_X_y(data: pd.DataFrame, label: str):
    X = data.drop(columns=label)
    y = data[label]
    return X, y


@task
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@task
def save_processed_data(data: dict, save_location: str):
    for name, df in data.items():
        df.to_pickle(f"{save_location}/{name}")


@flow(log_prints=True)
def process(
    data_location: DataLocation = DataLocation(),
    config: ProcessConfig = ProcessConfig(),
):
    data = get_data(data_location.raw_location, data_location.raw_file)
    processed = drop_columns(data, config.drop_columns)
    X, y = get_X_y(processed, config.label)
    split_data = split_train_test(X, y, config.test_size)
    save_processed_data(split_data, data_location.process_location)


if __name__ == "__main__":
    process()
