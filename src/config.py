from typing import List
from pydantic import BaseModel
from typing import Literal


class DataLocation(BaseModel):
    raw_location: Literal["data/raw", "data/processed"] = "data/raw"
    raw_file: str = "iris.csv"
    process_location: Literal["data/raw", "data/processed"] = "data/processed"


class ProcessConfig(BaseModel):
    drop_columns: list = ["Id"]
    label: str = "Species"
    test_size: float = 0.3


class SVC_Params(BaseModel):
    C: List[float] = [0.1, 1, 10, 100, 1000]
    gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]
