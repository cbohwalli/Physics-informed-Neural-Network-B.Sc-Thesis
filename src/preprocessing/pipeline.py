import pandas as pd
from src.preprocessing.standardize import standardize_per_profile
from src.preprocessing.ewma import add_ewma
from src.preprocessing.ewma import add_ewma

# Shared Thesis Constants
TARGET_COLUMNS = ["stator_yoke", "stator_winding", "stator_tooth", "pm"]
REDUNDANT_BASE = ["torque", "total_output_power", "residual_loss", "efficiency"]

def base_preprocess(df, input_features):
    """
    The core engine for data cleaning. 
    Standardizes and applies EWMA based on experiment requirements.
    """
    df = df.copy()

    # 1. Remove redundant features
    df.drop(columns=REDUNDANT_BASE, errors='ignore', inplace=True)

    # 2. Add EWMA (Temporal features)
    final_inputs = list(input_features)
    df = add_ewma(df, input_features)
    wma_features = [f"{col}_wma" for col in input_features]
    final_inputs += wma_features
    
    # 3. Standardize
    df = standardize_per_profile(df, final_inputs)

    return df, final_inputs

# --- Experiment Presets ---

def preprocess_fnn(df):
    """
    Experiment 2: FNN Benchmark.
    Uses EWMA and Standardization for optimal convergence.
    """
    # Dynamically identify inputs
    inputs = [c for c in df.columns if c not in TARGET_COLUMNS + ['timestamp', 'profile_id'] + REDUNDANT_BASE]
    
    return base_preprocess(df, inputs)

def preprocess_pinn(df):
    """
    Experiment 4: PINN (Hybrid).
    1. Standardizes data for the NN inputs (using the FNN preset).
    2. Attaches raw SI-unit features for the Physics Loss term.
    """
    # Step 1: Get the standardized FNN-style data
    processed_df, input_features = preprocess_fnn(df)

    # Step 2: Identify features required for the LPTN equations (Physics Loss)
    # These must be in raw units (Volts, Amps, Celsius, etc.)
    physics_features = [
        "copper_loss", "residual_loss", "i_q", "i_d", 
        "motor_speed", "ambient", "coolant"
    ]

    # Step 3: Map raw values from the original 'df' back to the 'processed_df'
    for col in physics_features:
        if col in df.columns:
            # Suffix '_raw' ensures the NN ignores these columns during training,
            # but the Loss Function can access them for physics verification.
            processed_df[f"{col}_raw"] = df[col].values

    return processed_df, input_features