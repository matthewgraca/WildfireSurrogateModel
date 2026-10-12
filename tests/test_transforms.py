import torch

from wildfire_simulator.transforms import MinMaxPerChannel

def test_transforms():
    min_vals = [7, 2, 9]
    max_vals = [47, 102, 41]
    transform = MinMaxPerChannel(min_vals, max_vals)

    x = torch.tensor([
        [
            [12, 35, 22],
            [19, 40, 11],
            [28, 15, 33]
        ],
        [
            [25, 14, 39],
            [31, 27, 18],
            [16, 30, 21]
        ],
        [
            [37, 20, 13],
            [24, 38, 17],
            [34, 29, 26]
        ]
    ], dtype=torch.float32)

    normalized = transform(x)

    expected = torch.tensor([
        [
            [0.1250, 0.7000, 0.3750],
            [0.3000, 0.8250, 0.1000],
            [0.5250, 0.2000, 0.6500]
        ],
        [
            [0.2300, 0.1200, 0.3700],
            [0.2900, 0.2500, 0.1600],
            [0.1400, 0.2800, 0.1900]
        ],
        [
            [0.8750, 0.3438, 0.1250],
            [0.4688, 0.9062, 0.2500],
            [0.7812, 0.6250, 0.5312]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(normalized, expected, atol=5e-5)
    assert torch.allclose(transform.inverse(normalized), x, atol=5e-5)
