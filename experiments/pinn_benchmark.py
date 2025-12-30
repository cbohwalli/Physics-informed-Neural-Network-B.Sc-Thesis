import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import GroupKFold

# Internal Imports
from src.preprocessing.pipeline import preprocess_pinn
from src.models.FNN.FNN_class import Feedforward_NN
from src.models.PINN.compute_physics_loss import compute_physics_loss
from src.models.PINN.train_utils import make_schedule

# --------------------------- Config ---------------------------
CONFIG = {
    "SEQUENCE_LENGTH": 60,
    "NUM_EPOCHS": 50,
    "BATCH_SIZE": 64,
    "EARLY_STOP_PATIENCE": 10,
    "DT_SECONDS": 0.5,
    "TARGET_COLUMNS": ["pm", "stator_yoke", "stator_winding", "stator_tooth"],
    "PHYSICS_COLS": ["residual_loss_raw", "copper_loss_raw", "i_q_raw", "i_d_raw", "motor_speed_raw", "ambient_raw", "coolant_raw"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Data Preparation ---------------------------
def prepare_pinn_tensors(df_processed, input_cols):
    """
    Transforms tabular data into windowed tensors for Hybrid PINN training.
    
    Returns:
        X: Model inputs (Standardized) - Shape: (N_windows, N_features)
        P: Physics sequences (Raw SI) - Shape: (N_windows, Seq_len, 6)
        Y: Target sequences - Shape: (N_windows, 4, Seq_len)
    """
    X_list, P_list, Y_list, groups_list = [], [], [], []
    
    for cycle_id, cycle_df in df_processed.groupby('profile_id'):
        num_timesteps = len(cycle_df)
        if num_timesteps < CONFIG["SEQUENCE_LENGTH"]: continue
        
        # Generate sliding window indices (e.g., 0-59, 1-60, etc.)
        indices = np.arange(num_timesteps - CONFIG["SEQUENCE_LENGTH"] + 1)[:, None] + np.arange(CONFIG["SEQUENCE_LENGTH"])
        
        # Standardized NN Inputs: Take the feature vector at the START of each window
        X_list.append(cycle_df[input_cols].values[indices[:, 0]])
        
        # Physics Sequences: Extract raw values for the ODE residual loss
        raw_vals = cycle_df[CONFIG["PHYSICS_COLS"]].values
        # Derived physical quantities: Current Magnitude (Amps) and Normalized Speed (0-1)
        I_mag = np.sqrt(raw_vals[:, 2]**2 + raw_vals[:, 3]**2) # sqrt(iq^2 + id^2)
        n_norm = raw_vals[:, 4] / 6000.0                       # Normalized against motor max RPM
        
        # Assemble physics tensor: [res_loss, copper_loss, I, n, ambient, coolant]
        p_seq = np.stack([raw_vals[:,0], raw_vals[:,1], I_mag, n_norm, raw_vals[:,5], raw_vals[:,6]], axis=1)
        P_list.append(p_seq[indices])
        
        # Targets: Temperatures reshaped to (Channels, Time) for Torch Conv/Loss
        # Transform from (N, Seq, 4) -> (N, 4, Seq)
        Y_list.append(np.transpose(cycle_df[CONFIG["TARGET_COLUMNS"]].values[indices], (0, 2, 1)))
        
        # Group ID for Cross-Validation 
        groups_list.extend([cycle_id] * len(indices))

    return (torch.tensor(np.vstack(X_list), dtype=torch.float32),
            torch.tensor(np.vstack(P_list), dtype=torch.float32),
            torch.tensor(np.vstack(Y_list), dtype=torch.float32),
            np.array(groups_list))

# --------------------------- Training Function ---------------------------
def run_cv_experiment(X, P, Y, groups):
    """
    Executes a 5-Fold Group Cross-Validation experiment using a specific 
    curriculum learning strategy to balance data-driven and physics-informed gradients.
    """
    print("Start Running PINN Experiment")
    kfold = GroupKFold(n_splits=5)
    fold_rmses = []
    fold_mae = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y, groups=groups)):
        # DataLoaders: pb (physics batch) is used for loss, not as model input
        train_loader = DataLoader(TensorDataset(X[train_idx], P[train_idx], Y[train_idx]), 
                                  batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(TensorDataset(X[val_idx], P[val_idx], Y[val_idx]), 
                                batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

        model = Feedforward_NN(input_dim=X.shape[1]).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        mse_crit = torch.nn.MSELoss()
        best_rmse = float('inf')

        for epoch in range(CONFIG["NUM_EPOCHS"]):
     
            model.train()
            epoch_data_loss = 0
            epoch_phys_loss = 0

            for xb, pb, yb in train_loader:
                xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
                optimizer.zero_grad()
                
                t0_batch = yb[:, :, 0] # Initial condition for the anchor

                # Forward Pass
                y_pred = model(xb, t0_batch)
                
                # Hybrid Loss Calculation:
                # 1. Data-driven Loss (MSE on targets, excluding t=0 initial condition)
                loss_data = mse_crit(y_pred[:,:,1:], yb[:,:,1:])
                # 2. Physics-informed Loss (ODE residual verification)
                loss_phy = compute_physics_loss(y_pred, pb)
                
                # Composite Gradient: Total Loss = λ_data*L_data + λ_phy*L_phy
                loss = (loss_data) + (loss_phy)
                
                loss.backward()
                optimizer.step()

                epoch_data_loss += loss_data.item()
                epoch_phys_loss += loss_phy.item()

            print(f"Epoch {epoch+1} | Data Loss: {epoch_data_loss/len(train_loader):.6f} | Physics Loss: {epoch_phys_loss/len(train_loader):.6f}")

            # Validation: Compute standard RMSE for performance tracking
            model.eval()
            val_errs = []
            with torch.no_grad():
                for xb, _, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    t0_batch = yb[:, :, 0] # Initial condition for the anchor

                    # Predict and slice to remove t=0
                    y_p = model(xb, t0_batch)[:, :, 1:]
                    y_t = yb[:, :, 1:]

                    val_errs.append((y_p - y_t).cpu().numpy())
            
            # Aggregate validation RMSE across all batches in the fold
            val_loss = np.mean(np.concatenate(val_errs)**2)
            val_rmse = np.sqrt(val_loss)
            val_mae = np.mean(np.abs(np.concatenate(val_errs)))

            scheduler.step(val_loss)

            # --- Early Stopping ---
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_mae = val_mae
                epochs_no_improve = 0

                # Save the best model for this fold
                model_path = f"results/saved_pinn/best_model_fold_{fold+1}.pt"
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rmse': best_rmse
                }, model_path)
                print(f"--> Saved best model for Fold {fold+1} at Epoch {epoch+1}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= CONFIG["EARLY_STOP_PATIENCE"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break 
        
        fold_rmses.append(best_rmse)
        fold_mae.append(best_mae)
        print(f"Fold {fold+1} RMSE: {best_rmse:.4f} | MAE: {best_mae:.4f}")

    return fold_rmses, fold_mae

# --------------------------- Main Loop ---------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/processed/dataset.csv")
    df_proc, input_cols = preprocess_pinn(df)
    X, P, Y, groups = prepare_pinn_tensors(df_proc, input_cols)

    # Run experiment and capture results
    rmses, maes = run_cv_experiment(X, P, Y, groups)

    # Create a summary DataFrame
    results_df = pd.DataFrame({
        "fold": np.arange(1, len(rmses) + 1),
        "rmse": rmses,
        "mae": maes
    })

    # Add a row for the average across all folds
    avg_row = pd.DataFrame([{
        "fold": "Average", 
        "rmse": np.mean(rmses), 
        "mae": np.mean(maes)
    }])
    
    results_df = pd.concat([results_df, avg_row], ignore_index=True)

    # Save to CSV
    results_df.to_csv("results/pinn_benchmark.csv", index=False)
    print("\nPINN experiments complete. Results saved to results/pinn_benchmark.csv")