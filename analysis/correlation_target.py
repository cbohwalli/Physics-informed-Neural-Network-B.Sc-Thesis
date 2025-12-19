import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load your data ---
df = pd.read_csv("dataset.csv")

# --- 2. Feature definitions ---
IRRELEVANT_FEATURES = ["timestamp", "profile_id"]
target_vars = ['stator_winding', 'stator_tooth', 'stator_yoke', 'pm']
input_vars = [col for col in df.columns if col not in IRRELEVANT_FEATURES + target_vars]

# --- 3. Correlation analysis ---
corr_df = df[input_vars + target_vars].corr()

print("\n=== Correlations between Inputs and Targets ===")
for target in target_vars:
    correlations = corr_df[target][input_vars]
    correlations_sorted = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    print(f"\nTarget: {target}")
    print(correlations_sorted)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(correlations_sorted.index, correlations_sorted.values, color='skyblue')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Correlations with {target}")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()