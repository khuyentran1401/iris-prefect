from prefect import flow

from process import process
from train_model import train


@flow
def main():
    process()
    train()


if __name__ == "__main__":
    main()
