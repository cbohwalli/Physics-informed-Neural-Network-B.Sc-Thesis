import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

# Internal Imports
from src.preprocessing.data_sequencing import create_output_sequence
from src.models.LPTN.lptn import lptn_simulate, split_residual

def main():
    print("--- Starting LPTN Physical Benchmark ---")
    
    # 1. Load Raw Data
    df = pd.read_csv("data/processed/dataset.csv")
    
    # Constants
    TARGET_COLUMNS = ["pm", "stator_yoke", "stator_winding", "stator_tooth"]
    INPUT_COLUMNS = [c for c in df.columns if c not in TARGET_COLUMNS + ["timestamp", "profile_id"]]
    
    # 2. Sequence Data for 60-step Trajectory
    print("Creating 60-step trajectory sequences...")
    dataset, groups, _ = create_output_sequence(
        df, 
        target_columns=TARGET_COLUMNS, 
        sequence_length=60,
        fully_sequenced=True
    )

    # 3. Setup Cross-Validation
    kfold = GroupKFold(n_splits=5)
    fold_summaries = []

    # Map column indices for fast lookup
    idx = {name: INPUT_COLUMNS.index(name) for name in [
        "copper_loss", "residual_loss", "i_q", "i_d", 
        "motor_speed", "ambient", "coolant"
    ]}

    # 4. Evaluation Loop
    X_tensor, Y_tensor = dataset.tensors
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor, Y_tensor, groups=groups)):
        print(f"\nEvaluating Fold {fold+1}/5...")
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

        rmse_list, mae_list = [], []

        for xb, yb in tqdm(val_loader, desc="Simulating Trajectories"):
            xb, yb = xb.squeeze(0), yb.squeeze(0)
            steps = xb.shape[0]

            # Signal Extraction
            Pw = xb[:, idx["copper_loss"]].numpy()
            P_res = xb[:, idx["residual_loss"]].numpy()
            I = np.sqrt(xb[:, idx["i_d"]].numpy()**2 + xb[:, idx["i_q"]].numpy()**2)
            speed_norm = xb[:, idx["motor_speed"]].numpy() / 6000.0
            
            Py, Pt, _ = split_residual(P_res, I, speed_norm)
            Ta = xb[:, idx["ambient"]].numpy()
            Tc = xb[:, idx["coolant"]].numpy()

            # Initial Conditions (Ground Truth at t=0)
            # TARGET_COLUMNS = ["pm", "stator_yoke", "stator_winding", "stator_tooth"]
            # yb shape is (4, 60). Indexing: yb[channel, time]
            T0 = (yb[1, 0].item(), yb[3, 0].item(), yb[2, 0].item(), yb[0, 0].item())

            # LPTN Simulation
            Ty_p, Tt_p, Tw_p, Tm_p = lptn_simulate(Py, Pt, Pw, Ta, Tc, T0, steps, dt=0.5)
            
            # 1. Stack predictions: (4, 60)
            # Order must match TARGET_COLUMNS: pm, yoke, winding, tooth
            y_pred = np.vstack([Tm_p, Ty_p, Tw_p, Tt_p])

            # 2. Slice both to remove the first timestep [:, 1:]
            y_true_eval = yb.numpy()[:, 1:]
            y_pred_eval = y_pred[:, 1:]

            # 3. Calculate errors on the sliced data
            errors = y_pred_eval - y_true_eval
            rmse_list.append(np.sqrt(np.mean(errors**2)))
            mae_list.append(np.mean(np.abs(errors)))

        # Store fold metrics
        fold_rmse = mean(rmse_list)
        fold_mae = mean(mae_list)
        fold_summaries.append({
            "fold": fold + 1,
            "rmse": fold_rmse,
            "mae": fold_mae
        })
        print(f"Fold {fold+1} Finished: RMSE={fold_rmse:.4f}")

    # 5. Final Statistics and CSV Export
    results_df = pd.DataFrame(fold_summaries)
    
    # Calculate Mean Row
    summary_row = pd.DataFrame([{
        "fold": "Average",
        "rmse": results_df["rmse"].mean(),
        "mae": results_df["mae"].mean()
    }])
    
    final_df = pd.concat([results_df, summary_row], ignore_index=True)
    
    # Save to CSV
    final_df.to_csv("results/lptn_benchmark_results.csv", index=False)
    
    print("\n" + "="*40)
    print(f"RESULTS SAVED TO: results/lptn_benchmark_results.csv")
    print(f"Final Avg RMSE: {results_df['rmse'].mean():.4f} °C")
    print("="*40)

if __name__ == "__main__":
    main()