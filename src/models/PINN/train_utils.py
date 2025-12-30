import math
import torch
import numpy as np

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