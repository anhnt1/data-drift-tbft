import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
import math
import argparse
import os
import itertools
import sys
from collections import defaultdict

# --- Corrected Helper Function for Cohen's d ---
def cohen_d(group1, group2):
    """Calculates Cohen's d for independent samples, expecting numpy arrays."""
    # Ensure inputs are numpy arrays and handle non-numeric if necessary (should be done before calling)
    # Convert to float to handle potential integer types before NaN checks
    group1 = np.asarray(group1, dtype=np.float64)
    group2 = np.asarray(group2, dtype=np.float64)

    # Remove NaNs using numpy
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        print(f"      Warning: Cannot compute Cohen's d with fewer than 2 valid samples in a group (Sizes after NaN removal: {len(group1)}, {len(group2)}).")
        return np.nan

    # Calculate the size of samples
    n1, n2 = len(group1), len(group2)
    # Calculate the variance of the samples
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Calculate the pooled standard deviation
    # Handle cases with zero variance carefully
    # Check for NaN variance as well (can happen if only one non-NaN value remains, though unlikely with len check)
    s1_is_zero_or_nan = (s1 == 0 or np.isnan(s1))
    s2_is_zero_or_nan = (s2 == 0 or np.isnan(s2))

    if s1_is_zero_or_nan and s2_is_zero_or_nan:
        pooled_std = 0
    elif s1_is_zero_or_nan:
        pooled_std = math.sqrt(s2) if not np.isnan(s2) else np.nan
    elif s2_is_zero_or_nan:
        pooled_std = math.sqrt(s1) if not np.isnan(s1) else np.nan
    else: # Standard case
        pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if np.isnan(pooled_std):
         print("      Warning: Pooled standard deviation is NaN.")
         return np.nan

    # Calculate the means of the samples
    u1, u2 = np.mean(group1), np.mean(group2)

    # Calculate the effect size, avoiding division by zero
    if pooled_std == 0:
        if u1 == u2:
            return 0.0
        else:
            print("      Warning: Cohen's d is undefined (infinite) due to zero pooled standard deviation but different means.")
            return np.inf
    return (u1 - u2) / pooled_std
# --- End Corrected Helper Function ---


# --- Main Analysis Function ---
def run_statistical_analysis(csv_filepath, alpha=0.05):
    """
    Loads the results CSV and performs statistical analysis comparing all methods.

    Args:
        csv_filepath (str): Path to the combined results CSV file.
        alpha (float): Significance level for tests.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: Input CSV file not found at '{csv_filepath}'")
        sys.exit(1)

    print(f"--- Loading Data from '{csv_filepath}' ---")
    try:
        df = pd.read_csv(csv_filepath)
        print(f"  Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # --- Data Cleaning and Preparation ---
    print("--- Preparing Data ---")
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['Accuracy'], inplace=True) # Drop rows with NaN accuracy early
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows due to non-numeric 'Accuracy' values.")
    if len(df) == 0:
        print("Error: No valid numeric accuracy data found after cleaning. Exiting.")
        sys.exit(1)

    ft_conditions_df = df[(df['RunIndex'] != -1) & (df['FTRatio'] != -1.0)].copy()
    conditions = ft_conditions_df[['Dataset', 'Model', 'DriftType', 'FTRatio']].drop_duplicates().values.tolist()
    print(f"Found {len(conditions)} unique conditions (Dataset, Model, DriftType, FTRatio) to analyze.")

    metric_mapping = {
        'resnet18': {
            'G1': 'GroupFT_conv1_layer1', 'G2': 'GroupFT_layer2',
            'G3': 'GroupFT_layer3', 'G4': 'GroupFT_layer4',
            'G5': 'GroupFT_fc', 'All': 'GroupFT_all'
        },
        'densenet121': {
            'G1': 'GroupFT_conv0_dense1', 'G2': 'GroupFT_dense2_trans2',
            'G3': 'GroupFT_dense3_trans3', 'G4': 'GroupFT_dense4_norm5',
            'G5': 'GroupFT_classifier', 'All': 'GroupFT_all'
        },
        'resnext50': {
            'G1': 'GroupFT_conv1_layer1', 'G2': 'GroupFT_layer2',
            'G3': 'GroupFT_layer3', 'G4': 'GroupFT_layer4',
            'G5': 'GroupFT_fc', 'All': 'GroupFT_all'
        }
    }
    baseline_metric_name = 'BaselineOnDrift'

    print(f"\n--- Starting Statistical Analysis (alpha = {alpha}) ---")

    for dataset, model, drift_type, ft_ratio in conditions:
        print(f"\n{'='*70}")
        print(f"Analyzing: Dataset={dataset}, Model={model}, DriftType={drift_type}, FTRatio={ft_ratio:.1f}")
        print(f"{'='*70}")

        # --- Data Extraction for this Condition ---
        condition_ft_df = ft_conditions_df[
            (ft_conditions_df['Dataset'] == dataset) &
            (ft_conditions_df['Model'] == model) &
            (ft_conditions_df['DriftType'] == drift_type) &
            (ft_conditions_df['FTRatio'] == ft_ratio)
        ]

        baseline_drift_df = df[
            (df['Dataset'] == dataset) &
            (df['Model'] == model) &
            (df['DriftType'] == drift_type) &
            (df['Metric'] == baseline_metric_name) &
            (df['RunIndex'] != -1)
        ]
        # Extract as numpy array after initial NaN drop
        baseline_drift_accuracies_np = baseline_drift_df['Accuracy'].values

        current_metric_map = metric_mapping.get(model)
        if not current_metric_map:
            print(f"  Warning: Metric mapping not found for model '{model}'. Cannot perform detailed group analysis.")
            continue

        analysis_groups_data = []
        analysis_group_names = []
        analysis_group_data_map = {} # Store numpy arrays for t-tests/power

        if len(baseline_drift_accuracies_np) >= 2:
            analysis_groups_data.append(baseline_drift_accuracies_np)
            analysis_group_names.append(baseline_metric_name)
            analysis_group_data_map[baseline_metric_name] = baseline_drift_accuracies_np
        else:
            print(f"  Warning: Skipping '{baseline_metric_name}' in analysis due to insufficient data points ({len(baseline_drift_accuracies_np)}).")

        available_ft_metrics = condition_ft_df['Metric'].unique()
        for generic_name, specific_metric in current_metric_map.items():
            if specific_metric in available_ft_metrics:
                # Extract as numpy array after initial NaN drop
                group_data_np = condition_ft_df[condition_ft_df['Metric'] == specific_metric]['Accuracy'].values
                if len(group_data_np) >= 2:
                    analysis_groups_data.append(group_data_np)
                    analysis_group_names.append(generic_name)
                    analysis_group_data_map[generic_name] = group_data_np
                else:
                    print(f"  Warning: Skipping metric '{specific_metric}' (as {generic_name}) in analysis due to insufficient data points ({len(group_data_np)}).")

        if len(analysis_groups_data) < 2:
            print("  Error: Need at least two groups with sufficient data for any comparison. Skipping condition.")
            continue

        # --- 1. ANOVA ---
        print("\n--- ANOVA: Comparing ALL Method Means (BaselineOnDrift + FT Groups) ---")
        if len(analysis_groups_data) >= 2:
            try:
                # Pass numpy arrays directly to f_oneway
                f_stat, p_value_anova = stats.f_oneway(*analysis_groups_data)
                print(f"  ANOVA Results (Groups: {', '.join(analysis_group_names)}):")
                print(f"    F-statistic: {f_stat:.4f}")
                print(f"    p-value:     {p_value_anova:.4g}")
                anova_significant = p_value_anova < alpha
                if anova_significant:
                    print("    Result: Significant difference detected between at least some method means (p < alpha).")
                else:
                    print("    Result: No significant difference detected between method means (p >= alpha).")
            except Exception as e_anova:
                print(f"  Error running ANOVA: {e_anova}")
                anova_significant = False
        else:
            print("  Not enough groups with sufficient data for ANOVA.")
            anova_significant = False

        # --- 2. Post-Hoc Test (Tukey HSD) ---
        if anova_significant:
            print("\n--- Tukey HSD Post-Hoc Test (All Pairwise Comparisons) ---")
            all_data_tukey = np.concatenate(analysis_groups_data)
            group_labels_tukey = np.concatenate([[name] * len(data) for name, data in zip(analysis_group_names, analysis_groups_data)])
            try:
                tukey_result = pairwise_tukeyhsd(all_data_tukey, group_labels_tukey, alpha=alpha)
                print(tukey_result)
            except Exception as e_tukey:
                print(f"    Error running Tukey HSD: {e_tukey}")
        elif len(analysis_groups_data) >= 2:
             print("\n--- Tukey HSD Post-Hoc Test ---")
             print("    Skipping Tukey HSD because ANOVA was not significant.")


        # --- 3. t-tests & 5. Power Analysis (Specific Pairs vs BaselineOnDrift) ---
        print("\n--- t-tests & Power Analysis: Comparing Each FT Method vs BaselineOnDrift ---")
        if baseline_metric_name not in analysis_group_names:
             print("  Cannot perform t-tests against baseline: BaselineOnDrift data was insufficient.")
        else:
            baseline_sample = analysis_group_data_map[baseline_metric_name] # Get the numpy array
            power_analyzer = TTestIndPower()

            for group_name in analysis_group_names:
                if group_name == baseline_metric_name:
                    continue

                print(f"\n  Comparison: {group_name} vs. {baseline_metric_name}")
                group_sample = analysis_group_data_map[group_name] # Get the numpy array

                # Perform t-test
                try:
                    # Pass numpy arrays directly to ttest_ind
                    t_stat, p_value_ttest = stats.ttest_ind(
                        group_sample,
                        baseline_sample,
                        equal_var=False, # Welch's t-test
                        nan_policy='omit' # Handles potential NaNs that might still exist if input was weird
                    )
                    print(f"    t-test Results:")
                    print(f"      t-statistic: {t_stat:.4f}")
                    print(f"      p-value:     {p_value_ttest:.4g}")
                    if p_value_ttest < alpha:
                        print("      Result: Significant difference detected (p < alpha).")
                    else:
                        print("      Result: No significant difference detected (p >= alpha).")

                    # Perform Power Analysis
                    # Pass numpy arrays directly to cohen_d
                    effect_size = cohen_d(group_sample, baseline_sample)
                    if not np.isnan(effect_size) and np.isfinite(effect_size):
                         nobs1 = min(len(group_sample), len(baseline_sample))
                         power = power_analyzer.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, alternative='two-sided')
                         print(f"    Post-Hoc Power Analysis:")
                         print(f"      Observed Effect Size (Cohen's d): {effect_size:.4f}")
                         print(f"      Statistical Power (for alpha={alpha}, N={nobs1}): {power:.4f}")
                    # else: # Already printed warning inside cohen_d if needed
                    #      print("    Could not calculate valid effect size for power analysis.")

                except Exception as e_ttest:
                    print(f"    Error running t-test/power analysis for {group_name}: {e_ttest}")

    print(f"\n{'='*70}")
    print("--- Statistical Analysis Script Finished ---")


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform statistical analysis on drift experiment results.')
    parser.add_argument(
        '--input_csv',
        type=str,
        default='./drift_results_combined/combined_drift_summary_runs30.csv', # Default path
        help='Path to the combined CSV file generated by drift_exp.py'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05).'
    )
    script_args = parser.parse_args()

    run_statistical_analysis(script_args.input_csv, script_args.alpha)