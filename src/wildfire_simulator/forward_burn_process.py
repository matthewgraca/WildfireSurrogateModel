import numpy as np
import torch

class ForwardBurnProcess:
    """
    Simulates forward burning up to a given time `t`.

    When called with a state frame (13×H×W) and a burn time `t`, it zeroes
    the fire-mask (channel 0) and arrival-time (channel 1) of every pixel
    whose original arrival time exceeds `t`.
    """
    def __call__(self, frame, t: float):
        not_burnt = frame[1] > t
        out = frame.clone()
        out[0][not_burnt] = 0.0
        out[1][not_burnt] = 0.0
        return out.to(frame.device, dtype=frame.dtype)
