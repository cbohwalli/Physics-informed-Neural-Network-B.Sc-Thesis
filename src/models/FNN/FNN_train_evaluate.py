import torch
import torch.nn as nn
import numpy as np

def validate(model, val_loader, criterion, device):
    """Helper to run validation and return core metrics."""
    model.eval()
    val_losses = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)

            # Slicing: drop first timestep 
            y_pred = y_pred[:, :, 1:]
            yb = yb[:, :, 1:]

            loss = criterion(y_pred, yb)
            val_losses.append(loss.item())
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Flatten for metric calculation
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    metrics = {
        "loss": np.mean(val_losses),
        "rmse": np.sqrt(np.mean((preds - targets) ** 2)),
        "mae": np.mean(np.abs(preds - targets)),
    }
    return metrics

def train_and_evaluate_model(model, train_loader, val_loader, fold, device, 
                             epochs=50, lr=1e-4, patience=10):
    """Main training loop with checkpointing and early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_metrics = {}
    early_stop_counter = 0
    model_path = f"best_model_fold_{fold + 1}.pt"

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation Phase ---
        val_results = validate(model, val_loader, criterion, device)
        avg_train_loss = np.mean(train_losses)

        # Update Learning Rate Scheduler
        scheduler.step(val_results["loss"])

        print(f"Fold {fold+1} | Epoch {epoch+1:02d}: Train Loss {avg_train_loss:.5f} | "
              f"Val Loss {val_results['loss']:.5f} | RMSE {val_results['rmse']:.5f}")

        # --- Checkpointing & Early Stopping ---
        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_metrics = val_results
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"--> Early stopping triggered at epoch {epoch+1}")
            break

    # Load the best state before returning
    model.load_state_dict(torch.load(model_path))
    return best_metrics["rmse"], best_metrics["mae"]