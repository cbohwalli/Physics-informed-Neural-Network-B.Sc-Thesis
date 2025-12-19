import torch.nn as nn

# ---------------------------
# Model definition
# ---------------------------
class Feedforward_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=4, output_dim=4*120, activation="ReLU"):
        super().__init__()
        
        act_fn = getattr(nn, activation)()
        layers = [nn.Linear(input_dim, hidden_dim), act_fn]

        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), 4, 120)