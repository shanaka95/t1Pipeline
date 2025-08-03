"""
Create processed data for Top1 severity prediction
This script creates a processed dataset similar to depression_processed_top5.csv 
but using action class features from the top1 dataset.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

def create_processed_top1_data():
    """Create processed top1 data for severity prediction"""
    print("Creating processed Top1 data for severity prediction...")
    print("="*60)
    
    # Load the top1 dataset
    df = pd.read_csv('../datasets/ml_depression_dataset_top1.csv')
    print(f"Loaded top1 dataset: {df.shape}")
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Extract action class features (action_class_00 to action_class_51)
    action_features = [col for col in df.columns if col.startswith('action_class_')]
    print(f"Found {len(action_features)} action class features")
    
    # Create feature info similar to top5
    feature_info = {
        'action_class_columns': action_features,
        'action_class_scaled_columns': [f"{col}_scaled" for col in action_features],
        'target_columns': ['Depression_Binary', 'Depression_3Class', 'depressed'],
        'patient_column': 'Patient_ID',
        'video_column': 'video_name'
    }
    
    # Scale the action class features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[action_features])
    
    # Add scaled features to the dataframe
    for i, col in enumerate(action_features):
        processed_df[f"{col}_scaled"] = scaled_features[:, i]
    
    # Add some engineered features similar to top5
    # Total activity across all action classes
    processed_df['total_action_activity'] = df[action_features].sum(axis=1)
    
    # Most active action class (index of max value)
    processed_df['most_active_action'] = df[action_features].idxmax(axis=1).str.replace('action_class_', '').astype(int)
    
    # Number of active action classes (non-zero)
    processed_df['num_active_actions'] = (df[action_features] > 0).sum(axis=1)
    
    # Action diversity (Shannon entropy-like measure)
    def calculate_diversity(row):
        values = row[action_features].values
        values = values[values > 0]  # Only non-zero values
        if len(values) <= 1:
            return 0
        # Normalize to probabilities
        probs = values / values.sum()
        # Calculate entropy using math.log
        import math
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
        return entropy
    
    processed_df['action_diversity'] = processed_df.apply(calculate_diversity, axis=1)
    
    # Update feature info with engineered features
    feature_info['engineered_columns'] = ['total_action_activity', 'most_active_action', 'num_active_actions', 'action_diversity']
    feature_info['all_feature_columns'] = action_features + feature_info['action_class_scaled_columns'] + feature_info['engineered_columns']
    
    # Select final columns for the processed dataset
    final_columns = (
        ['Patient_ID'] + 
        feature_info['target_columns'] + 
        action_features + 
        feature_info['action_class_scaled_columns'] + 
        feature_info['engineered_columns']
    )
    
    # Create final processed dataframe
    final_df = processed_df[final_columns].copy()
    
    print(f"\nProcessed dataset shape: {final_df.shape}")
    print(f"Features breakdown:")
    print(f"  - Action class features: {len(action_features)}")
    print(f"  - Scaled action class features: {len(action_features)}")
    print(f"  - Engineered features: {len(feature_info['engineered_columns'])}")
    print(f"  - Target columns: {len(feature_info['target_columns'])}")
    print(f"  - Total features: {len(feature_info['all_feature_columns'])}")
    
    # Show target distribution
    print(f"\nTarget Distribution:")
    print(f"Depression_Binary distribution:")
    print(final_df['Depression_Binary'].value_counts().sort_index())
    print(f"\nDepression_3Class distribution:")
    print(final_df['Depression_3Class'].value_counts().sort_index())
    
    # Calculate class balance
    binary_counts = final_df['Depression_Binary'].value_counts()
    severity_counts = final_df['Depression_3Class'].value_counts()
    
    binary_balance = binary_counts.max() / binary_counts.min() if binary_counts.min() > 0 else "N/A"
    severity_balance = severity_counts.max() / severity_counts.min() if severity_counts.min() > 0 else "N/A"
    
    print(f"\nClass Balance Ratios:")
    print(f"  Binary classification: {binary_balance:.2f}" if binary_balance != "N/A" else "  Binary classification: N/A")
    print(f"  3-class severity: {severity_balance:.2f}" if severity_balance != "N/A" else "  3-class severity: N/A")
    
    if severity_balance != "N/A" and severity_balance > 3:
        print("  ⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED - SMOTE will be beneficial")
    
    # Save processed data
    output_file = '../processed_data/depression_processed_top1.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Processed data saved to: {output_file}")
    
    # Save feature info
    feature_info_file = '../processed_data/top1_feature_info.pkl'
    with open(feature_info_file, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"✅ Feature info saved to: {feature_info_file}")
    
    # Save scaler
    scaler_file = '../processed_data/top1_scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to: {scaler_file}")
    
    print(f"\n🎯 TOP1 PROCESSED DATA READY FOR SEVERITY PREDICTION!")
    print(f"Use these files for severity prediction models:")
    print(f"  - Data: {output_file}")
    print(f"  - Feature info: {feature_info_file}")
    print(f"  - Scaler: {scaler_file}")
    
    return final_df, feature_info

if __name__ == "__main__":
    processed_df, feature_info = create_processed_top1_data()