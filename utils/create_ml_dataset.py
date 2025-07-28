import pandas as pd
import numpy as np

def create_ml_ready_dataset():
    """
    Create a final ML-ready dataset for training depression classification models
    """
    
    print("=== CREATING ML-READY DATASET ===")
    
    # Read the final depression dataset
    df = pd.read_csv('video_cluster_with_final_depression.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Depression status distribution:")
    print(df['Binary_Depression'].value_counts())
    
    # Remove patients without questionnaire data (no labels for training)
    df_ml = df[df['Binary_Depression'] != 'No Questionnaire Data'].copy()
    
    print(f"\nAfter removing patients without labels: {df_ml.shape}")
    print(f"Remaining depression distribution:")
    print(df_ml['Binary_Depression'].value_counts())
    
    # Create binary target variable for ML
    # 1 = Depressed (including moderate depression), 0 = Not Depressed/Mild
    df_ml['Depression_Binary'] = df_ml['Binary_Depression'].apply(
        lambda x: 1 if 'Depressed' in x and 'Mild' not in x else 0
    )
    
    # Create a 3-class target for multi-class classification if needed
    df_ml['Depression_3Class'] = df_ml['Binary_Depression'].apply(
        lambda x: 2 if 'Depressed' in x and 'Mild' not in x 
                 else 1 if 'Mild' in x 
                 else 0
    )
    
    # Select relevant columns for ML
    # Features: All cluster percentages
    cluster_cols = [col for col in df_ml.columns if col.startswith('cluster_')]
    
    # Metadata columns to keep
    metadata_cols = [
        'Patient ID',
        'Patient_ID',
        'Depression_Binary',  # Main binary target
        'Depression_3Class',  # 3-class target
        'Binary_Depression',  # Original classification
        'Overall_Depression_Status',
        'Overall_Severity',
        'Confidence',
        'PHQ9_Score',
        'HRSD_Score',
        'ADS_Score',
        'SKID_Depressed'
    ]
    
    # Combine features and metadata
    final_cols = ['Patient_ID'] + cluster_cols + [
        'Depression_Binary',
        'Depression_3Class', 
        'Binary_Depression',
        'Overall_Depression_Status',
        'Confidence',
        'PHQ9_Score',
        'HRSD_Score', 
        'ADS_Score',
        'SKID_Depressed'
    ]
    
    # Create final ML dataset
    df_final = df_ml[final_cols].copy()
    
    # Handle any remaining missing values in features
    print(f"\nMissing values in cluster features:")
    cluster_missing = df_final[cluster_cols].isnull().sum().sum()
    print(f"Total missing cluster values: {cluster_missing}")
    
    if cluster_missing > 0:
        # Fill missing cluster values with 0 (assuming missing = no activity in that cluster)
        df_final[cluster_cols] = df_final[cluster_cols].fillna(0)
        print("Filled missing cluster values with 0")
    
    # Verify data quality
    print(f"\n=== FINAL DATASET SUMMARY ===")
    print(f"Shape: {df_final.shape}")
    print(f"Features (clusters): {len(cluster_cols)}")
    print(f"Patients: {len(df_final)}")
    
    print(f"\nBinary target distribution:")
    binary_dist = df_final['Depression_Binary'].value_counts()
    print(f"Not Depressed (0): {binary_dist[0]} ({binary_dist[0]/len(df_final)*100:.1f}%)")
    print(f"Depressed (1): {binary_dist[1]} ({binary_dist[1]/len(df_final)*100:.1f}%)")
    
    print(f"\n3-Class target distribution:")
    class3_dist = df_final['Depression_3Class'].value_counts().sort_index()
    for i, count in class3_dist.items():
        label = ['Not Depressed', 'Mild/Subclinical', 'Depressed'][i]
        print(f"{label} ({i}): {count} ({count/len(df_final)*100:.1f}%)")
    
    print(f"\nConfidence distribution:")
    conf_dist = df_final['Confidence'].value_counts()
    print(conf_dist)
    
    # Data quality checks
    print(f"\n=== DATA QUALITY CHECKS ===")
    
    # Check for duplicate patients
    duplicates = df_final['Patient_ID'].duplicated().sum()
    print(f"Duplicate patients: {duplicates}")
    
    # Check cluster sum (should be close to 1.0 for each patient)
    cluster_sums = df_final[cluster_cols].sum(axis=1)
    print(f"Cluster percentage sums - Min: {cluster_sums.min():.3f}, Max: {cluster_sums.max():.3f}, Mean: {cluster_sums.mean():.3f}")
    
    # Check for any infinite or extreme values
    inf_values = np.isinf(df_final[cluster_cols].values).sum()
    print(f"Infinite values in clusters: {inf_values}")
    
    extreme_values = (df_final[cluster_cols] > 1.0).sum().sum()
    print(f"Cluster values > 1.0: {extreme_values}")
    
    # Save the final ML dataset
    output_file = 'ml_depression_dataset.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ Final ML dataset saved as: {output_file}")
    
    # Create a features-only file for some ML workflows
    features_file = 'ml_features_only.csv'
    df_features = df_final[['Patient_ID'] + cluster_cols].copy()
    df_features.to_csv(features_file, index=False)
    print(f"✅ Features-only dataset saved as: {features_file}")
    
    # Create a targets-only file
    targets_file = 'ml_targets_only.csv'  
    df_targets = df_final[['Patient_ID', 'Depression_Binary', 'Depression_3Class', 'Binary_Depression']].copy()
    df_targets.to_csv(targets_file, index=False)
    print(f"✅ Targets-only dataset saved as: {targets_file}")
    
    # Create a metadata summary
    metadata_file = 'ml_metadata.csv'
    df_metadata = df_final[['Patient_ID', 'Binary_Depression', 'Overall_Depression_Status', 
                           'Confidence', 'PHQ9_Score', 'HRSD_Score', 'ADS_Score', 'SKID_Depressed']].copy()
    df_metadata.to_csv(metadata_file, index=False)
    print(f"✅ Metadata saved as: {metadata_file}")
    
    return df_final

def cleanup_intermediate_files():
    """
    Remove unnecessary intermediate files
    """
    import os
    
    print(f"\n=== CLEANING UP INTERMEDIATE FILES ===")
    
    files_to_remove = [
        'depression_analysis_summary.csv',
        'video_cluster_with_depression.csv', 
        'final_depression_classification_summary.csv',
        'video_cluster_with_final_depression.csv',
        'analyze_questionnaires.py',
        'create_depression_analysis.py',
        'validate_hrsd_logic.py',
        'final_depression_classification.py',
        'create_ml_dataset.py'  # Remove this script too after execution
    ]
    
    removed_count = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file}: {e}")
        else:
            print(f"⚠️ File not found: {file}")
    
    print(f"\n🗑️ Removed {removed_count} intermediate files")
    
    # List remaining files
    print(f"\n📁 Remaining files:")
    for file in os.listdir('.'):
        if file.endswith('.csv') or file.endswith('.py') or file.endswith('.xlsx'):
            size = os.path.getsize(file) / 1024  # Size in KB
            print(f"  {file} ({size:.1f} KB)")

if __name__ == "__main__":
    # Create final ML dataset
    df_final = create_ml_ready_dataset()
    
    # Clean up intermediate files
    cleanup_intermediate_files()
    
    print(f"\n🎯 FINAL ML DATASET READY!")
    print(f"Main file: ml_depression_dataset.csv")
    print(f"Features: 100 cluster percentages")
    print(f"Binary target: Depression_Binary (0=Not Depressed, 1=Depressed)")
    print(f"3-class target: Depression_3Class (0=Not Depressed, 1=Mild, 2=Depressed)")
    print(f"Patients: {len(df_final)}") 