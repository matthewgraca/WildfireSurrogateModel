import torch

from wildfire_simulator.datasets import WildfireDataset
from wildfire_simulator.forward_burn_process import ForwardBurnProcess

def test_forward_burn_process(dataloader):
    dataset = WildfireDataset(dataloader)
    burner = ForwardBurnProcess()

    tensor = dataset[0]

    new_tensor = burner(tensor, 30)
    tensor_expected = torch.load("tests/baseline/burned_item_0.pt")
    assert (new_tensor == tensor_expected).all()

