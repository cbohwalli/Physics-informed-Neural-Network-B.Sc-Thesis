import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

# Custom Imports from your project structure
from src.preprocessing.pipeline import preprocess_fnn
from src.preprocessing.data_sequencing import create_output_sequence
from src.models.FNN.FNN_class import Feedforward_NN
from src.models.FNN.FNN_train_evaluate import train_and_evaluate_model

# Constants for the Experiment
K_FOLDS = 5
BATCH_SIZE = 64

def run_cross_validation(dataset, groups, input_dim, config, device):
    """
    Handles the K-Fold logic for a single hyperparameter configuration.
    """
    kfold = GroupKFold(n_splits=K_FOLDS)
    fold_metrics = []
    
    # Extract tensors for the split
    X_data = dataset.tensors[0]
    Y_data = dataset.tensors[1]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_data, Y_data, groups=groups)):
        print(f"    Fold {fold+1}/{K_FOLDS}...", end="\r")
        
        # Create DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize Model
        model = Feedforward_NN(
            input_dim=input_dim, 
            hidden_dim=config['hidden_dim'], 
            num_layers=config['num_layers'], 
            activation=config['activation']
        ).to(device)
        
        # Train and Evaluate (returns RMSE, MAE, MaxAE)
        metrics = train_and_evaluate_model(model, train_loader, val_loader, fold, device)
        fold_metrics.append(metrics)

    # Return the average metrics across all folds
    return np.mean(fold_metrics, axis=0)

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting FNN Benchmark Experiment on: {device}")

    # 1. Load and Prepare Data 
    print("Preparing dataset...")

    df = pd.read_csv("data/processed/dataset.csv")
    # Removes redudant features, Adds EMWA and standardize input features
    df_processed_final , final_input_cols = preprocess_fnn(df)
    # Create Sequences: This strictly handles the 120-step output windowing
    dataset, groups, input_dim = create_output_sequence(
    df_processed_final, 
    target_columns=["stator_yoke", "stator_winding", "stator_tooth", "pm"],
    sequence_length=120
    )

    print(f"Data ready. Input Dimension: {input_dim}")

    # 2. Define the Full Hyperparameter Grid
    test_configs = [
        # ReLU Configurations
        {"hidden_dim": 128, "num_layers": 4, "activation": "ReLU"},
        {"hidden_dim": 128, "num_layers": 8, "activation": "ReLU"},
        {"hidden_dim": 128, "num_layers": 12, "activation": "ReLU"}, 
        {"hidden_dim": 512, "num_layers": 4, "activation": "ReLU"}, 
        {"hidden_dim": 512, "num_layers": 8, "activation": "ReLU"}, 
        {"hidden_dim": 512, "num_layers": 12, "activation": "ReLU"}, 
        {"hidden_dim": 1024, "num_layers": 4, "activation": "ReLU"},
        # SiLU (Swish) Configurations
        {"hidden_dim": 128, "num_layers": 4, "activation": "SiLU"},
        {"hidden_dim": 128, "num_layers": 8, "activation": "SiLU"},
        {"hidden_dim": 128, "num_layers": 12, "activation": "SiLU"}, 
        {"hidden_dim": 512, "num_layers": 4, "activation": "SiLU"}, 
        {"hidden_dim": 512, "num_layers": 8, "activation": "SiLU"}, 
        {"hidden_dim": 512, "num_layers": 12, "activation": "SiLU"}, 
        {"hidden_dim": 1024, "num_layers": 4, "activation": "SiLU"}
    ]

    # 3. Execution Loop
    all_results = []
    
    for i, cfg in enumerate(test_configs):
        print(f"\n[Config {i+1}/{len(test_configs)}] Testing: {cfg}")
        
        # Run Cross-Validation for this config
        avg_metrics = run_cross_validation(dataset, groups, input_dim, cfg, device)
        
        # Store results
        result_entry = {
            **cfg, 
            "Avg_RMSE": avg_metrics[0], 
            "Avg_MAE": avg_metrics[1],
        }
        all_results.append(result_entry)

    # 4. Final Export for Thesis
    results_df = pd.DataFrame(all_results)
    final_output_path = "results/fnn_benchmark_final.csv"
    results_df.sort_values(by="Avg_RMSE", ascending=True).to_csv(final_output_path, index=False)
    
    print("\n" + "="*30)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {final_output_path}")
    print("="*30)

if __name__ == "__main__":
    main()