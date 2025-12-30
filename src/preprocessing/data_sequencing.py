import torch
import numpy as np
import pandas as pd
import gc  # Garbage Collector for manual memory management
from torch.utils.data import TensorDataset

def create_output_sequence(df, target_columns, sequence_length=60, fully_sequenced=False):
    """
    Memory-efficient sequencing of tabular motor data.
    
    Args:
        df (pd.DataFrame): The preprocessed dataset.
        target_columns (list): Thermal targets (e.g., stator_winding, pm).
        sequence_length (int): Trajectory window size.
        fully_sequenced (bool): If True, returns (Windows, Seq, Features) for LPTN.
                               If False, returns (Windows, Features) for standard FNN.
    """
    exclude = target_columns + ['timestamp', 'profile_id']
    input_cols = [c for c in df.columns if c not in exclude]
    
    X_list, Y_list, groups_list = [], [], []

    # Sequential Windowing
    for profile_id, group_df in df.groupby('profile_id'):
        # Force float32 immediately to prevent NumPy's float64 memory bloat
        cycle_X = group_df[input_cols].values.astype(np.float32)
        cycle_y = group_df[target_columns].values.astype(np.float32)
        
        num_timesteps = len(group_df)
        if num_timesteps < sequence_length:
            continue

        indices = np.arange(num_timesteps - sequence_length + 1)[:, None] + np.arange(sequence_length)
        
        if fully_sequenced:
            x_seq = cycle_X[indices] # Shape: (Num_windows, 60, Num_features)
            # Guard: If only 1 window is generated, ensure it stays 3D
            if x_seq.ndim == 2:
                x_seq = x_seq[np.newaxis, ...]
            X_list.append(x_seq)
        else:
            X_list.append(cycle_X[indices[:, 0]]) # Shape: (Num_windows, Num_features)

        # Transpose to (Num_windows, Channels, Seq) 
        y_seq = np.transpose(cycle_y[indices], (0, 2, 1))
        Y_list.append(y_seq)
        
        groups_list.extend([profile_id] * len(indices))

    # Concatenate the lists into large NumPy arrays
    X_np = np.concatenate(X_list, axis=0)
    X_list.clear() 
    
    Y_np = np.concatenate(Y_list, axis=0)
    Y_list.clear() 
    
    # Force Garbage Collection to reclaim RAM before Tensor creation
    gc.collect()

    # Tensor Conversion
    X_tensor = torch.from_numpy(X_np)
    Y_tensor = torch.from_numpy(Y_np)
    
    groups = np.array(groups_list)
    
    return TensorDataset(X_tensor, Y_tensor), groups, len(input_cols)