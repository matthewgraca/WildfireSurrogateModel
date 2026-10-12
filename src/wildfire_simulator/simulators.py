import torch
import copy
import numpy as np
import torch.nn.functional as F

def fire_burn_step(t, model, inputs, dt=None):
    inputs = copy.deepcopy(inputs)
    inputs = torch.cat((inputs, torch.full((1, 1, 500, 500), t, device=inputs.device)), dim=1)
    inputs = F.pad(inputs, (6, 6, 6, 6, 0, 0), mode='constant', value=0)
    with torch.no_grad():
        pred = model(inputs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        # pred shape: (1, C, H, W) where C=1 for mask-only model
        # Threshold mask to binary
        pred_mask = (pred[0, 0].detach() > 0.5).float()
        # Update mask in unpadded region
        inputs[0, 0, 6:-6, 6:-6] = pred_mask[6:-6, 6:-6]
        # Deterministic FAT update: newly burned pixels get arrival time = t + dt
        if dt is not None:
            current_fat = inputs[0, 1, 6:-6, 6:-6]
            newly_burned = (pred_mask[6:-6, 6:-6] == 1) & (current_fat == 0)
            inputs[0, 1, 6:-6, 6:-6][newly_burned] = t + dt
    return inputs[:, :13, 6:-6, 6:-6]


class ForwardBurnSimulator:
    def __init__(
        self,
        data,
        model,
        step,
        transform,
        dt,
        max_t,
        t0=0
    ):
        self.data = data
        self.model = model
        self.step = step
        self.transform = transform
        self.dt = dt
        self.max_t = max_t
        self.t0 = t0

    def run_to(self, t, return_history=False):
        input = self.transform(self.data)

        # Apply burn process at t0: zero channels 0-1 where arrival time > t0
        # (matches training, where the model never sees the full unmasked input)
        if isinstance(input, torch.Tensor) and input.dim() == 4:
            t0_normalized = self.t0 / self.max_t
            not_burnt = input[:, 1:2, :, :] > t0_normalized
            input = input.clone()
            input[:, 0:1][not_burnt] = 0.0
            input[:, 1:2][not_burnt] = 0.0
            history = [self.transform.inverse(input)]
        else:
            history = [self.data]
        dt = self.dt/self.max_t
        for i in np.arange(self.t0/self.max_t, t/self.max_t, dt):
            try:
                input = self.step(i, self.model, input, dt=dt)
            except TypeError:
                input = self.step(i, self.model, input)
            history.append(self.transform.inverse(input))
        return history if return_history else history[-1]
