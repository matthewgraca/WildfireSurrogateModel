import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wildfire_simulator.losses import DistanceMapLoss


def make_block_target(n=1, size=64, block=24):
    """(n, 1, size, size) binary target with a centered block mask."""
    target = torch.zeros(n, 1, size, size)
    start = (size - block) // 2
    target[:, 0, start:start + block, start:start + block] = 1.0
    return target


def test_distance_maps_values():
    loss = DistanceMapLoss()
    target = make_block_target(n=2, size=64, block=24)
    d = loss._distance_maps(target)

    assert d.shape == target.shape
    assert torch.isfinite(d).all()
    assert d.min() >= 0.0 and d.max() <= 1.0
    d = d[0, 0].numpy()

    # 24x24 block at rows/cols 20..43
    boundary = d[20, 20:44]      # interior edge pixels
    boundary_out = d[19, 20:44]  # exterior edge pixels
    deep_inside = d[32, 32]
    deep_outside = d[0, 0]

    # Boundary pixels: weight 1 - dist/max on each side, normalized by that
    # side's own max. Interior side: max interior distance is 12 (half the
    # block). Exterior side: max exterior distance is the corner-to-block
    # distance sqrt(20^2 + 20^2).
    expected_in = 1.0 - 1.0 / 12.0
    expected_out = 1.0 - 1.0 / np.sqrt(800.0)
    assert np.allclose(boundary, expected_in, atol=1e-5)
    assert np.allclose(boundary_out, expected_out, atol=1e-5)
    # Deep interior / far exterior pixels have no penalty weight
    assert deep_inside < 1e-6
    assert deep_outside < 1e-6


def test_degenerate_targets_reduce_to_plain_bce():
    """All-background and all-foreground targets give d = 0, so the loss
    must equal the unweighted binary cross-entropy."""
    loss = DistanceMapLoss()
    torch.manual_seed(0)
    pred = 0.05 + 0.9 * torch.rand(2, 1, 32, 32)

    for target in (torch.zeros(2, 1, 32, 32), torch.ones(2, 1, 32, 32)):
        with torch.no_grad():
            expected = F.binary_cross_entropy(pred, target)
        assert torch.allclose(loss._distance_maps(target),
                               torch.zeros_like(target))
        assert torch.allclose(loss(pred, target), expected)


def test_boundary_errors_penalized_more_than_interior_errors():
    """Identical per-pixel errors cost more at the fire front than deep
    inside the burned region."""
    loss = DistanceMapLoss()
    target = make_block_target(n=1, size=64, block=24)

    # Confidently wrong prediction over the whole block: nll is identical at
    # every target pixel, only the (1 + d) weight differs.
    pred = torch.full_like(target, 0.05)
    with torch.no_grad():
        plain_bce = F.binary_cross_entropy(pred, target)
        dml = loss(pred, target)
    assert dml > plain_bce

    # And the unweighted term logged matches plain BCE exactly
    assert loss.last_ce == pytest.approx(plain_bce.item(), abs=1e-6)
    assert loss.last_penalty > 0.0
    # L = mean(nll) + mean(d * nll)
    assert dml == pytest.approx(loss.last_ce + loss.last_penalty, abs=1e-6)


def test_backward_flows_to_pred():
    loss = DistanceMapLoss()
    target = make_block_target(n=2, size=32, block=16)
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # No gradient leaks into the (constant) distance-map path
    assert target.grad is None
