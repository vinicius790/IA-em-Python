import pytest
from main import load_dataset, prepare_data

def test_load_dataset():
    dataset = load_dataset()
    assert dataset is not None
    assert not dataset.empty

def test_prepare_data():
    dataset = load_dataset()
    X_train, X_validation, Y_train, Y_validation = prepare_data(dataset)
    assert X_train.shape[0] > 0
    assert Y_train.shape[0] > 0

if __name__ == "__main__":
    pytest.main()
