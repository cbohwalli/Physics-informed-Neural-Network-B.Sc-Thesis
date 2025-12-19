import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_per_profile(df, input_features, profile_col="profile_id"):
    """
    Standardizes input features independently for each profile (drive cycle)
    and returns the updated DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Your dataset containing input features and profile IDs.
    input_features : list of str
        Columns to standardize.
    profile_col : str
        Column identifying the drive cycle
    
    Returns:
    --------
    df_standardized : pd.DataFrame
        DataFrame with standardized input features (other columns unchanged).
    """
    
    df_standardized = df.copy()

    # Apply standardization group by group
    for profile, group in df.groupby(profile_col):
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(group[input_features])
        df_standardized.loc[group.index, input_features] = scaled_values

    return df_standardized