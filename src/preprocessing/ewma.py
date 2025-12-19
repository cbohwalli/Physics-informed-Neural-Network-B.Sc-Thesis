def add_ewma(df, feature_cols, group_col='profile_id', alpha=0.001):
    """
    Applies exponential weighted moving average (EWMA) smoothing per drive cycle
    for the given feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the features and group column.
    feature_cols : list of str
        Columns to apply smoothing to.
    group_col : str
        Column name used to group data
    alpha : float
        Smoothing factor for EWMA.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added '_wma' columns for each smoothed feature.
    """
    df = df.copy()

    for col in feature_cols:
        df[f'{col}_wma'] = (
            df.groupby(group_col)[col]
            .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )

    return df