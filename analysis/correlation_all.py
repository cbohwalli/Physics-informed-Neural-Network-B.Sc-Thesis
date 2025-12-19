import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

IRRELEVANT_FEATURES = ["timestamp", "profile_id"]

# ---------------------------
# 1. Load and prepare dataset
# ---------------------------
df = pd.read_csv("dataset.csv")  
df = df.drop(columns=IRRELEVANT_FEATURES)

# ---------------------------
# 2. Compute correlation matrix
# ---------------------------
corr_matrix = df.corr(method='pearson')

# ---------------------------
# 3. Create sorted correlation table for each feature
# ---------------------------
def sorted_correlations(corr_matrix, threshold=None):
    """
    Prints each feature and its correlation with others, sorted by absolute value.
    If threshold is set, only show correlations above that value.
    """
    for col in corr_matrix.columns:
        corrs = corr_matrix[col].drop(col)  # remove self-correlation
        corrs_sorted = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
        
        if threshold:
            corrs_sorted = corrs_sorted[abs(corrs_sorted) >= threshold]
            if corrs_sorted.empty:
                continue
        
        print(f"\nFeature: {col}")
        print(corrs_sorted.to_string())

# ---------------------------
# 4. Highlight highly correlated pairs
# ---------------------------
threshold = 0.9
print(f"Highly correlated features (abs(corr) >= {threshold}):")
sorted_correlations(corr_matrix, threshold=threshold)

# ---------------------------
# 5. Visualize correlation heatmap
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    annot_kws={"size":9}
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()