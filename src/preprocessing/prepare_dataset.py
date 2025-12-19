import pandas as pd
import numpy as np

def add_losses(df, Rs0=0.0141, alpha_Cu=0.0039, beta_Cu1=0.315, beta_Cu2=0.616, n_max=None):
    """
    Compute and add copper and residual (iron+mechanical) losses to dataset.

    Parameters
    ----------
    df : pd.DataFrame
    Rs0 : float
        Base stator resistance at reference temperature (Ohms)
    alpha_Cu : float
        Temperature coefficient for copper (~0.0039 1/K)
    beta_Cu1, beta_Cu2 : floats
        Speed correction coefficients
    n_max : float
        Maximum speed (rpm) for normalization. If None, taken as max in dataset.
    """
    
    df = df.copy()

    if n_max is None:
        n_max = df["motor_speed"].max()

    # --- Instantaneous stator current magnitude ---
    df["i_inst"] = np.sqrt(df["i_d"]**2 + df["i_q"]**2)
    # --- Temperature correction factor ---
    T_ref = 25  # °C reference
    df["f1_temp"] = 1 + alpha_Cu * (df["stator_winding"] - T_ref)
    # --- Speed correction factor ---
    df["f2_speed"] = 1 + beta_Cu1 * (df["motor_speed"] / n_max) + beta_Cu2 * (df["motor_speed"] / n_max) ** 2
    # --- Effective stator resistance ---
    df["R_s"] = Rs0 * df["f1_temp"] * df["f2_speed"]
    # --- Copper losses per sample ---
    df["copper_loss"] = 3 * (df["i_inst"] ** 2) * df["R_s"]

    # --- Residual losses (eg. iron + mechanical) ---
    if "total_power_loss" in df.columns:
        df["residual_loss"] = df["total_power_loss"] - df["copper_loss"]
        df["residual_loss"] = df["residual_loss"].clip(lower=0)
    else:
        print("Warning: 'total_power_loss' not found. Only copper losses computed.")

    # --- cleanup ---
    df.drop(columns=["i_inst", "f1_temp", "f2_speed", "R_s"], inplace=True)

    return df

def add_features(df):
    df = df.copy()

    # --- Calculate total electrical input power ---
    df["total_input_power"] = 1.5 * (df["u_d"] * df["i_d"] + df["u_q"] * df["i_q"])
    # --- Calculate mechanical power ---
    df["total_output_power"] = df["torque"] * (2 * np.pi * df["motor_speed"] / 60)
    # --- Calculate efficiency ---
    df["efficiency"] = np.where(df["total_input_power"] != 0, df["total_output_power"] / df["total_input_power"], np.nan)
    # --- Calculate total losses  ---
    df["total_power_loss"] = df["total_input_power"] - df["total_output_power"]
    # --- Add timestamps ---
    time_step = 0.5  # seconds
    df['timestamp'] = df.groupby('profile_id').cumcount() * time_step
    # --- Add copper and residual losses ---
    df = add_losses(df)

    return df

if __name__ == "__main__":
    df = pd.read_csv("measures_v2.csv")
    df = add_features(df)
    df.to_csv('dataset.csv', index=False)