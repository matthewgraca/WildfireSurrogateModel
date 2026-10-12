import torch

class MinMaxPerChannel:
    def __init__(self, min_vals, max_vals):
        self.min_vals = torch.as_tensor(min_vals, dtype=torch.float32)
        self.max_vals = torch.as_tensor(max_vals, dtype=torch.float32)

    def __call__(self, img):
        return self.transform(img)

    def _get_params(self, img):
        C = img.shape[-3]
        if img.dim() == 3:
            min_vals = self.min_vals.to(img.device).view(C, 1, 1)
            max_vals = self.max_vals.to(img.device).view(C, 1, 1)
        else:
            min_vals = self.min_vals.to(img.device).view(1, C, 1, 1)
            max_vals = self.max_vals.to(img.device).view(1, C, 1, 1)

        range_vals = max_vals - min_vals
        range_vals = torch.clamp(range_vals, min=1e-8)
        return min_vals, range_vals

    def transform(self, img):
        min_vals, range_vals = self._get_params(img)
        return (img - min_vals) / range_vals
    
    def inverse(self, img):
        min_vals, range_vals = self._get_params(img)
        return img * range_vals + min_vals


