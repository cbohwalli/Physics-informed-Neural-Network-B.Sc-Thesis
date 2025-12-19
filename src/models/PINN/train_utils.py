import math
import torch
import numpy as np

def make_schedule(method="linear", epochs=50, start=0.0, end=1.0, sharpness=10.0):
    """Creates a function that returns a weight factor based on current epoch."""
    def linear_schedule(e):
        t = e / float(epochs - 1)
        return float(start + (end - start) * t)
    
    def sigmoid_schedule(e):
        t = e / float(epochs - 1)
        x = (t - 0.5) * sharpness
        s = 1.0 / (1.0 + math.exp(-x))
        return float(start + (end - start) * s)
    
    return linear_schedule if method == "linear" else sigmoid_schedule

def compute_gradient_feature_importance(model, loader, device, input_names):
    """Estimates feature importance via gradients of the output w.r.t inputs."""
    model.eval()
    grads_accum = None
    n_samples = 0

    for xb, pb, yb in loader:
        xb = xb.to(device).requires_grad_(True)
        y_pred = model(xb)
        loss = y_pred.mean()
        
        grads = torch.autograd.grad(loss, xb)[0]
        grads_abs = grads.abs().detach().cpu().numpy()

        if grads_accum is None:
            grads_accum = grads_abs.sum(axis=0)
        else:
            grads_accum += grads_abs.sum(axis=0)
        n_samples += grads_abs.shape[0]

    importance = (grads_accum / n_samples)
    importance /= (np.sum(importance) + 1e-8)
    return dict(zip(input_names, importance.tolist()))