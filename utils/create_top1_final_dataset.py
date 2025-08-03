#!/usr/bin/env python3
"""
Create a final ML-ready dataset for top1 action class approach.
Combines video action class percentages with depression data.
"""

import pandas as pd
import numpy as np
import os

def load_action_class_data(action_class_file):
    """Load video action class percentages."""
    print(f"Loading action class data from: {action_class_file}")
    df = pd.read_csv(action_class_file)
    
    # Extract patient ID from video name (assuming format like "XXX_t1_YYYYMMDD")
    df['Patient_ID'] = df['video_name'].str.extract(r'(\d+)_t1_')[0].astype(int)
    
    print(f"Loaded action class data for {len(df)} videos")
    print(f"Unique patients: {df['Patient_ID'].nunique()}")
    
    return df

def load_depression_data(depression_file):
    """Load depression status data."""
    print(f"Loading depression data from: {depression_file}")
    df = pd.read_csv(depression_file)
    
    # Rename 'id' column to 'Patient_ID' for consistency
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'Patient_ID'})
    
    print(f"Loaded depression data for {len(df)} patients")
    
    return df

def merge_datasets(action_df, depression_df):
    """Merge action class percentages with depression data."""
    print("Merging datasets...")
    
    # Merge on Patient_ID
    merged_df = pd.merge(action_df, depression_df, on='Patient_ID', how='inner')
    
    print(f"Merged dataset contains {len(merged_df)} records")
    print(f"Patients with both video and depression data: {merged_df['Patient_ID'].nunique()}")
    
    return merged_df

def create_depression_targets(df):
    """Create different depression target variables for ML."""
    
    # Binary target: 1 = depressed, 0 = not depressed
    df['Depression_Binary'] = df['depressed'].astype(int)
    
    # Multi-class target based on depression level
    level_mapping = {
        'Minimal': 0,
        'Mild': 1, 
        'Moderate': 2,
        'Severe': 3
    }
    df['Depression_Level_Numeric'] = df['depression_level'].map(level_mapping)
    
    # 3-class simplified: 0=None/Minimal, 1=Mild, 2=Moderate/Severe
    df['Depression_3Class'] = df['Depression_Level_Numeric'].apply(
        lambda x: 0 if x == 0 else (1 if x == 1 else 2)
    )
    
    return df

def prepare_final_dataset(merged_df):
    """Prepare the final ML dataset with proper feature selection."""
    
    # Action class features
    action_features = [col for col in merged_df.columns if col.startswith('action_class_')]
    
    # Select final columns
    final_cols = ['Patient_ID', 'video_name'] + action_features + [
        'Depression_Binary',
        'Depression_3Class', 
        'Depression_Level_Numeric',
        'depression_level',  # Original level
        'depressed'  # Original binary
    ]
    
    final_df = merged_df[final_cols].copy()
    
    # Handle any missing values in action class features
    print(f"Missing values in action class features:")
    action_missing = final_df[action_features].isnull().sum().sum()
    print(f"Total missing action class values: {action_missing}")
    
    if action_missing > 0:
        # Fill missing values with 0 (no activity in that action class)
        final_df[action_features] = final_df[action_features].fillna(0)
        print("Filled missing action class values with 0")
    
    return final_df, action_features

def display_dataset_summary(final_df, action_features):
    """Display comprehensive dataset summary."""
    print(f"\n=== FINAL TOP1 DATASET SUMMARY ===")
    print(f"Shape: {final_df.shape}")
    print(f"Features (action classes): {len(action_features)}")
    print(f"Patients: {final_df['Patient_ID'].nunique()}")
    print(f"Videos: {len(final_df)}")
    
    # Target distributions
    print(f"\nBinary target distribution:")
    binary_dist = final_df['Depression_Binary'].value_counts()
    for value, count in binary_dist.items():
        label = 'Depressed' if value == 1 else 'Not Depressed'
        print(f"  {label} ({value}): {count} ({count/len(final_df)*100:.1f}%)")
    
    print(f"\n3-Class target distribution:")
    class3_dist = final_df['Depression_3Class'].value_counts().sort_index()
    class_labels = ['None/Minimal', 'Mild', 'Moderate/Severe']
    for value, count in class3_dist.items():
        label = class_labels[value]
        print(f"  {label} ({value}): {count} ({count/len(final_df)*100:.1f}%)")
    
    print(f"\nDetailed depression level distribution:")
    level_dist = final_df['depression_level'].value_counts()
    for level, count in level_dist.items():
        print(f"  {level}: {count} ({count/len(final_df)*100:.1f}%)")
    
    # Data quality checks
    print(f"\n=== DATA QUALITY CHECKS ===")
    
    # Check for duplicate patients
    duplicates = final_df['Patient_ID'].duplicated().sum()
    print(f"Duplicate patients: {duplicates}")
    
    # Check action class sum (should be close to 1.0 for each video)
    action_sums = final_df[action_features].sum(axis=1)
    print(f"Action class percentage sums - Min: {action_sums.min():.3f}, Max: {action_sums.max():.3f}, Mean: {action_sums.mean():.3f}")
    
    # Check for any infinite or extreme values
    inf_values = np.isinf(final_df[action_features].values).sum()
    print(f"Infinite values in action classes: {inf_values}")
    
    extreme_values = (final_df[action_features] > 1.0).sum().sum()
    print(f"Action class values > 1.0: {extreme_values}")
    
    # Show most active action classes
    action_means = final_df[action_features].mean().sort_values(ascending=False)
    print(f"\nTop 10 most active action classes (by mean percentage):")
    for i, (action_class, mean_pct) in enumerate(action_means.head(10).items()):
        class_id = action_class.split('_')[-1]
        print(f"  {i+1:2d}. {action_class} (Class {class_id}): {mean_pct:.4f} ({mean_pct*100:.2f}%)")

def save_datasets(final_df, action_features):
    """Save the final datasets in multiple formats."""
    
    # Main dataset
    main_file = 'ml_depression_dataset_top1.csv'
    final_df.to_csv(main_file, index=False)
    print(f"\n✅ Main dataset saved as: {main_file}")
    
    # Features-only file
    features_file = 'ml_features_only_top1.csv'
    df_features = final_df[['Patient_ID', 'video_name'] + action_features].copy()
    df_features.to_csv(features_file, index=False)
    print(f"✅ Features-only dataset saved as: {features_file}")
    
    # Targets-only file
    targets_file = 'ml_targets_only_top1.csv'  
    df_targets = final_df[['Patient_ID', 'video_name', 'Depression_Binary', 'Depression_3Class', 
                          'Depression_Level_Numeric', 'depression_level', 'depressed']].copy()
    df_targets.to_csv(targets_file, index=False)
    print(f"✅ Targets-only dataset saved as: {targets_file}")
    
    # Patient-level aggregated data (in case of multiple videos per patient)
    patient_agg = final_df.groupby('Patient_ID').agg({
        **{col: 'mean' for col in action_features},  # Average action class percentages
        'Depression_Binary': 'first',  # Depression status should be same for all videos of a patient
        'Depression_3Class': 'first',
        'Depression_Level_Numeric': 'first',
        'depression_level': 'first',
        'depressed': 'first',
        'video_name': 'count'  # Count videos per patient
    }).reset_index()
    
    # Rename the count column
    patient_agg = patient_agg.rename(columns={'video_name': 'num_videos'})
    
    patient_file = 'ml_patient_level_top1.csv'
    patient_agg.to_csv(patient_file, index=False)
    print(f"✅ Patient-level dataset saved as: {patient_file}")
    print(f"    Aggregated {len(final_df)} videos from {len(patient_agg)} patients")
    print(f"    Average videos per patient: {final_df.groupby('Patient_ID').size().mean():.1f}")

def main():
    """Main execution function."""
    
    # File paths
    action_class_file = 'video_action_class_percentages_top1.csv'
    depression_file = 'datasets/depression_status_summary.csv'
    
    # Check if input files exist
    if not os.path.exists(action_class_file):
        print(f"❌ Error: Action class file not found: {action_class_file}")
        print("Please run create_top1_percentage_csv.py first to generate action class percentages.")
        return
    
    if not os.path.exists(depression_file):
        print(f"❌ Error: Depression file not found: {depression_file}")
        return
    
    # Load datasets
    action_df = load_action_class_data(action_class_file)
    depression_df = load_depression_data(depression_file)
    
    # Merge datasets
    merged_df = merge_datasets(action_df, depression_df)
    
    if len(merged_df) == 0:
        print("❌ Error: No matching records found between action class and depression data")
        return
    
    # Create depression targets
    merged_df = create_depression_targets(merged_df)
    
    # Prepare final dataset
    final_df, action_features = prepare_final_dataset(merged_df)
    
    # Display summary
    display_dataset_summary(final_df, action_features)
    
    # Save datasets
    save_datasets(final_df, action_features)
    
    print(f"\n🎯 TOP1 ACTION CLASS DATASET READY!")
    print(f"Main file: ml_depression_dataset_top1.csv")
    print(f"Features: 52 action class percentages (0-51)")
    print(f"Binary target: Depression_Binary (0=Not Depressed, 1=Depressed)")
    print(f"3-class target: Depression_3Class (0=None/Minimal, 1=Mild, 2=Moderate/Severe)")
    print(f"Videos: {len(final_df)}")
    print(f"Patients: {final_df['Patient_ID'].nunique()}")

if __name__ == "__main__":
    main()