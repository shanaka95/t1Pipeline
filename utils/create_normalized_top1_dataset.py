"""
Create normalized version of Top1 action class dataset for fair comparison with Top5.
This script applies StandardScaler normalization to action class features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_normalized_top1_dataset():
    """Create normalized top1 dataset similar to top5 scaled features"""
    print("Creating Normalized Top1 Action Class Dataset")
    print("="*60)
    
    # Load original top1 dataset
    original_path = 'datasets/ml_depression_dataset_top1.csv'
    df = pd.read_csv(original_path)
    
    print(f"Original dataset loaded: {df.shape}")
    
    # Identify action class features
    action_features = [col for col in df.columns if col.startswith('action_class_')]
    print(f"Action class features: {len(action_features)}")
    
    # Analyze original data
    print(f"\nOriginal Data Statistics:")
    action_data = df[action_features]
    print(f"  Min: {action_data.min().min():.6f}")
    print(f"  Max: {action_data.max().max():.6f}")
    print(f"  Mean: {action_data.mean().mean():.6f}")
    print(f"  Std: {action_data.std().mean():.6f}")
    
    # Create normalized version
    scaler = StandardScaler()
    action_data_scaled = scaler.fit_transform(action_data)
    
    # Create scaled column names
    scaled_feature_names = [f"{col}_scaled" for col in action_features]
    
    # Create new dataframe with both original and scaled features
    df_normalized = df.copy()
    
    # Add scaled features
    scaled_df = pd.DataFrame(action_data_scaled, columns=scaled_feature_names, index=df.index)
    df_normalized = pd.concat([df_normalized, scaled_df], axis=1)
    
    print(f"\nNormalized Data Statistics:")
    print(f"  Min: {action_data_scaled.min():.6f}")
    print(f"  Max: {action_data_scaled.max():.6f}")
    print(f"  Mean: {action_data_scaled.mean():.6f}")
    print(f"  Std: {action_data_scaled.std():.6f}")
    
    # Save normalized dataset
    output_path = 'datasets/ml_depression_dataset_top1_normalized.csv'
    df_normalized.to_csv(output_path, index=False)
    print(f"\nNormalized dataset saved: {output_path}")
    print(f"New dataset shape: {df_normalized.shape}")
    
    # Save scaler for future use
    scaler_path = 'datasets/top1_action_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")
    
    # Create feature info similar to top5
    feature_info = {
        'action_class_features': action_features,
        'action_class_scaled_features': scaled_feature_names,
        'patient_features': ['Patient_ID'],
        'video_features': ['video_name'],
        'target_features': ['Depression_Binary', 'Depression_3Class', 'Depression_Level_Numeric', 'depression_level', 'depressed']
    }
    
    feature_info_path = 'datasets/top1_feature_info.pkl'
    with open(feature_info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"Feature info saved: {feature_info_path}")
    
    # Display sample data comparison
    print(f"\nSample Data Comparison:")
    print(f"Original action_class_10 (first 5 values): {df['action_class_10'].head().values}")
    print(f"Scaled action_class_10 (first 5 values): {df_normalized['action_class_10_scaled'].head().values}")
    
    # Show feature distributions
    print(f"\nFeature Distribution Analysis:")
    print(f"Original features with mean > 0.1: {(action_data.mean() > 0.1).sum()}")
    print(f"Original features with mean > 0.05: {(action_data.mean() > 0.05).sum()}")
    print(f"Original features with mean > 0.01: {(action_data.mean() > 0.01).sum()}")
    
    scaled_means = pd.Series(action_data_scaled.mean(axis=0), index=action_features)
    print(f"Scaled features with |mean| > 0.1: {(np.abs(scaled_means) > 0.1).sum()}")
    print(f"Scaled features with |mean| > 0.05: {(np.abs(scaled_means) > 0.05).sum()}")
    
    return df_normalized, scaler, feature_info

if __name__ == "__main__":
    df_normalized, scaler, feature_info = create_normalized_top1_dataset()
    
    print(f"\n✅ Normalized Top1 dataset created successfully!")
    print(f"📊 Ready for fair comparison with Top5 scaled features")
    print(f"🎯 Both approaches now use standardized features")