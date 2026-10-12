import numpy as np
import torch
from torch.utils.data import Dataset

from wildfire_simulator.datasets import WildfireDataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel

def test_dataset(dataloader):
    dataset = WildfireDataset(dataloader)

    assert len(dataset) == 2
    
    tensor = dataset[0]
    tensor_expected = torch.load("tests/baseline/dataset_item_0.pt")
    assert (tensor == tensor_expected).all()

def test_min_max_norm(dataloader):
    dataset = WildfireDataset(dataloader)

    min_val = dataset.min_val
    min_val_expected = np.load("tests/baseline/min.npy")
    assert (min_val == min_val_expected).all()

    max_val = dataset.max_val
    max_val_expected = np.load("tests/baseline/max.npy")
    assert (max_val == max_val_expected).all()

def test_transformed_dataset(dataloader):
    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    transformed_dataset = TransformedDataset(dataset, transform)

    tensor = transformed_dataset[1]
    tensor_expected = torch.load("tests/baseline/normed_item_1.pt")
    assert (tensor == tensor_expected).all()

