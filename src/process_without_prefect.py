import pandas as pd
from sklearn.model_selection import train_test_split


def get_raw_data(data_location: str, file_name: str):
    return pd.read_csv(f"{data_location}/{file_name}")


def drop_columns(data: pd.DataFrame, columns: list):
    return data.drop(columns=columns)


def get_X_y(data: pd.DataFrame, label: str):
    X = data.drop(columns=label)
    y = data[label]
    return X, y


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


def save_processed_data(data: dict, save_location: str):
    for name, df in data.items():
        df.to_pickle(f"{save_location}/{name}")


def process(
    raw_location: str = "data/raw",
    process_location: str = "data/processed",
    raw_file: str = "iris.csv",
    label: str = "Species",
    test_size: float = 0.3,
    columns_to_drop=["Id"],
):
    data = get_raw_data(raw_location, raw_file)
    processed = drop_columns(data, columns=columns_to_drop)
    X, y = get_X_y(processed, label)
    split_data = split_train_test(X, y, test_size)
    save_processed_data(split_data, process_location)


if __name__ == "__main__":
    process(test_size=0.4, columns_to_drop=["col1", "col2"])
