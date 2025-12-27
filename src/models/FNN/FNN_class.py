import torch
import torch.nn as nn

# ---------------------------
# Model definition
# ---------------------------
class Feedforward_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=12, output_dim=4*60, activation="SiLU"):
        super().__init__()
        
        act_fn = getattr(nn, activation)()
        layers = [nn.Linear(input_dim, hidden_dim), act_fn]

        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t0):
        """
        x: [Batch, 60, features]
        t0: [Batch, 4] -> The actual temperatures at step 0 of the window
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Raw prediction from NN
        out = self.net(x_flat).view(batch_size, 4, 60)

        # 1. Create the time-mask forcing term
        # t goes from 0 to 1 over 60 steps
        t = torch.linspace(0, 1, 60).to(x.device).view(1, 1, 60)

        # This factor is 0.0 at t=0. 
        forcing_term = (1 - torch.exp(-10 * t)) 

        # 2. Apply Hard Constraint: Prediction = Initial + (NN_Adjustment * Mask)
        # Result: At index 0, result is ALWAYS exactly t0.
        y_constrained = t0.unsqueeze(-1) + (out * forcing_term)

        return y_constrained