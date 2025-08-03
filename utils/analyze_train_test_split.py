"""
Comprehensive Analysis of Train-Test Split Methodology
This script analyzes the train-test split to ensure proper ML practices:
1. No data leakage (same patients in train/test)
2. Proper data separation
3. Test set balance analysis
4. Distribution comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_dataset():
    """Load the top1 dataset and analyze its structure"""
    print("Loading Top1 Action Class Dataset...")
    df = pd.read_csv('datasets/ml_depression_dataset_top1.csv')
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Videos: {len(df)}")
    print(f"Unique Patients: {df['Patient_ID'].nunique()}")
    
    # Check for multiple videos per patient
    videos_per_patient = df.groupby('Patient_ID').size()
    print(f"\nVideos per Patient:")
    print(f"  - Min: {videos_per_patient.min()}")
    print(f"  - Max: {videos_per_patient.max()}")
    print(f"  - Mean: {videos_per_patient.mean():.2f}")
    print(f"  - Patients with multiple videos: {(videos_per_patient > 1).sum()}")
    
    if (videos_per_patient > 1).sum() > 0:
        print(f"\n⚠️  WARNING: Some patients have multiple videos!")
        print(f"This could lead to data leakage if not handled properly.")
        
        multi_video_patients = videos_per_patient[videos_per_patient > 1]
        print(f"Patients with multiple videos: {len(multi_video_patients)}")
        print(f"Examples: {list(multi_video_patients.head().index)}")
    
    return df

def analyze_current_split_method(df):
    """Analyze how the current train-test split works"""
    print(f"\n{'='*60}")
    print("CURRENT SPLIT METHOD ANALYSIS")
    print(f"{'='*60}")
    
    # Simulate the current split method (video-level split)
    action_features = [col for col in df.columns if col.startswith('action_class_')]
    X = df[action_features]
    y = df['Depression_Binary']
    
    # Current method: video-level split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get the indices to map back to original data
    train_indices = X_train.index
    test_indices = X_test.index
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    print(f"Current Method: Video-level split")
    print(f"Train set: {len(train_df)} videos from {train_df['Patient_ID'].nunique()} patients")
    print(f"Test set: {len(test_df)} videos from {test_df['Patient_ID'].nunique()} patients")
    
    # Check for patient overlap
    train_patients = set(train_df['Patient_ID'])
    test_patients = set(test_df['Patient_ID'])
    overlap = train_patients.intersection(test_patients)
    
    print(f"\n🔍 Data Leakage Check:")
    print(f"  - Patients in train: {len(train_patients)}")
    print(f"  - Patients in test: {len(test_patients)}")
    print(f"  - Overlapping patients: {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"  ⚠️  DATA LEAKAGE DETECTED! {len(overlap)} patients appear in both sets")
        print(f"  Overlapping patients: {list(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
        
        # Analyze overlap details
        overlap_videos = df[df['Patient_ID'].isin(overlap)]
        print(f"  Total videos from overlapping patients: {len(overlap_videos)}")
        print(f"  These patients contribute to both training and testing!")
    else:
        print(f"  ✅ No data leakage: patients are properly separated")
    
    return train_df, test_df, overlap

def analyze_target_balance(train_df, test_df):
    """Analyze target distribution in train and test sets"""
    print(f"\n{'='*60}")
    print("TARGET DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    # Binary target analysis
    print(f"Binary Depression Distribution:")
    
    train_binary = train_df['Depression_Binary'].value_counts()
    test_binary = test_df['Depression_Binary'].value_counts()
    
    print(f"\nTrain Set:")
    for value, count in train_binary.items():
        label = 'Depressed' if value == 1 else 'Not Depressed'
        print(f"  {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print(f"\nTest Set:")
    for value, count in test_binary.items():
        label = 'Depressed' if value == 1 else 'Not Depressed'
        print(f"  {label}: {count} ({count/len(test_df)*100:.1f}%)")
    
    # Calculate balance metrics
    train_balance = train_binary.min() / train_binary.max()
    test_balance = test_binary.min() / test_binary.max()
    
    print(f"\nBalance Ratio (minority/majority):")
    print(f"  Train set balance: {train_balance:.3f} ({'Balanced' if train_balance > 0.8 else 'Imbalanced'})")
    print(f"  Test set balance: {test_balance:.3f} ({'Balanced' if test_balance > 0.8 else 'Imbalanced'})")
    
    # 3-class analysis
    print(f"\n3-Class Depression Distribution:")
    
    train_3class = train_df['Depression_3Class'].value_counts().sort_index()
    test_3class = test_df['Depression_3Class'].value_counts().sort_index()
    
    class_labels = ['None/Minimal', 'Mild', 'Moderate/Severe']
    
    print(f"\nTrain Set (3-class):")
    for value, count in train_3class.items():
        print(f"  {class_labels[value]}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print(f"\nTest Set (3-class):")
    for value, count in test_3class.items():
        print(f"  {class_labels[value]}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return train_binary, test_binary, train_3class, test_3class

def suggest_proper_split_method(df):
    """Suggest a proper patient-level split to avoid data leakage"""
    print(f"\n{'='*60}")
    print("RECOMMENDED SPLIT METHOD")
    print(f"{'='*60}")
    
    # Patient-level split
    unique_patients = df['Patient_ID'].unique()
    patient_labels = df.groupby('Patient_ID')['Depression_Binary'].first()
    
    # Split patients, not videos
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42, 
        stratify=patient_labels
    )
    
    # Get corresponding videos
    proper_train_df = df[df['Patient_ID'].isin(train_patients)]
    proper_test_df = df[df['Patient_ID'].isin(test_patients)]
    
    print(f"Recommended Method: Patient-level split")
    print(f"Train set: {len(proper_train_df)} videos from {len(train_patients)} patients")
    print(f"Test set: {len(proper_test_df)} videos from {len(test_patients)} patients")
    
    # Verify no overlap
    assert set(train_patients).intersection(set(test_patients)) == set(), "Patient overlap detected!"
    print(f"✅ No patient overlap: proper data separation")
    
    # Check balance
    print(f"\nProper Split Target Distribution:")
    
    proper_train_binary = proper_train_df['Depression_Binary'].value_counts()
    proper_test_binary = proper_test_df['Depression_Binary'].value_counts()
    
    print(f"\nTrain Set:")
    for value, count in proper_train_binary.items():
        label = 'Depressed' if value == 1 else 'Not Depressed'
        print(f"  {label}: {count} ({count/len(proper_train_df)*100:.1f}%)")
    
    print(f"\nTest Set:")
    for value, count in proper_test_binary.items():
        label = 'Depressed' if value == 1 else 'Not Depressed'
        print(f"  {label}: {count} ({count/len(proper_test_df)*100:.1f}%)")
    
    return proper_train_df, proper_test_df

def create_visualization(df, train_df, test_df, proper_train_df, proper_test_df):
    """Create comprehensive visualizations comparing split methods"""
    print(f"\nCreating Split Comparison Visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Train-Test Split Analysis: Current vs Recommended Methods', fontsize=16, fontweight='bold')
    
    # 1. Dataset overview
    ax = axes[0, 0]
    videos_per_patient = df.groupby('Patient_ID').size()
    ax.hist(videos_per_patient, bins=range(1, videos_per_patient.max()+2), alpha=0.7, edgecolor='black')
    ax.set_title('Videos per Patient Distribution')
    ax.set_xlabel('Number of Videos')
    ax.set_ylabel('Number of Patients')
    ax.grid(True, alpha=0.3)
    
    # 2. Current split - target distribution
    ax = axes[0, 1]
    current_data = [
        ['Train', 'Not Depressed', (train_df['Depression_Binary'] == 0).sum()],
        ['Train', 'Depressed', (train_df['Depression_Binary'] == 1).sum()],
        ['Test', 'Not Depressed', (test_df['Depression_Binary'] == 0).sum()],
        ['Test', 'Depressed', (test_df['Depression_Binary'] == 1).sum()]
    ]
    current_df_plot = pd.DataFrame(current_data, columns=['Split', 'Class', 'Count'])
    sns.barplot(data=current_df_plot, x='Split', y='Count', hue='Class', ax=ax)
    ax.set_title('Current Split: Target Distribution')
    ax.legend(title='Depression Status')
    
    # 3. Recommended split - target distribution
    ax = axes[0, 2]
    proper_data = [
        ['Train', 'Not Depressed', (proper_train_df['Depression_Binary'] == 0).sum()],
        ['Train', 'Depressed', (proper_train_df['Depression_Binary'] == 1).sum()],
        ['Test', 'Not Depressed', (proper_test_df['Depression_Binary'] == 0).sum()],
        ['Test', 'Depressed', (proper_test_df['Depression_Binary'] == 1).sum()]
    ]
    proper_df_plot = pd.DataFrame(proper_data, columns=['Split', 'Class', 'Count'])
    sns.barplot(data=proper_df_plot, x='Split', y='Count', hue='Class', ax=ax)
    ax.set_title('Recommended Split: Target Distribution')
    ax.legend(title='Depression Status')
    
    # 4. Patient distribution comparison
    ax = axes[1, 0]
    split_comparison = pd.DataFrame({
        'Method': ['Current (Video-level)', 'Recommended (Patient-level)'],
        'Train_Patients': [train_df['Patient_ID'].nunique(), proper_train_df['Patient_ID'].nunique()],
        'Test_Patients': [test_df['Patient_ID'].nunique(), proper_test_df['Patient_ID'].nunique()]
    })
    
    x = range(len(split_comparison))
    width = 0.35
    ax.bar([i - width/2 for i in x], split_comparison['Train_Patients'], width, label='Train', alpha=0.8)
    ax.bar([i + width/2 for i in x], split_comparison['Test_Patients'], width, label='Test', alpha=0.8)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Patient Distribution by Split Method')
    ax.set_xticks(x)
    ax.set_xticklabels(split_comparison['Method'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Balance comparison
    ax = axes[1, 1]
    
    current_train_balance = train_df['Depression_Binary'].value_counts().min() / train_df['Depression_Binary'].value_counts().max()
    current_test_balance = test_df['Depression_Binary'].value_counts().min() / test_df['Depression_Binary'].value_counts().max()
    proper_train_balance = proper_train_df['Depression_Binary'].value_counts().min() / proper_train_df['Depression_Binary'].value_counts().max()
    proper_test_balance = proper_test_df['Depression_Binary'].value_counts().min() / proper_test_df['Depression_Binary'].value_counts().max()
    
    balance_data = pd.DataFrame({
        'Split_Method': ['Current', 'Current', 'Recommended', 'Recommended'],
        'Set_Type': ['Train', 'Test', 'Train', 'Test'],
        'Balance_Ratio': [current_train_balance, current_test_balance, proper_train_balance, proper_test_balance]
    })
    
    sns.barplot(data=balance_data, x='Split_Method', y='Balance_Ratio', hue='Set_Type', ax=ax)
    ax.set_title('Class Balance Comparison')
    ax.set_ylabel('Balance Ratio (min/max)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Balance Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Data leakage visualization
    ax = axes[1, 2]
    
    # Check overlaps
    train_patients_current = set(train_df['Patient_ID'])
    test_patients_current = set(test_df['Patient_ID'])
    overlap_current = len(train_patients_current.intersection(test_patients_current))
    
    train_patients_proper = set(proper_train_df['Patient_ID'])
    test_patients_proper = set(proper_test_df['Patient_ID'])
    overlap_proper = len(train_patients_proper.intersection(test_patients_proper))
    
    leakage_data = pd.DataFrame({
        'Method': ['Current (Video-level)', 'Recommended (Patient-level)'],
        'Overlapping_Patients': [overlap_current, overlap_proper]
    })
    
    bars = ax.bar(leakage_data['Method'], leakage_data['Overlapping_Patients'], 
                  color=['red' if x > 0 else 'green' for x in leakage_data['Overlapping_Patients']], alpha=0.7)
    ax.set_title('Data Leakage: Overlapping Patients')
    ax.set_ylabel('Number of Overlapping Patients')
    ax.set_xticklabels(leakage_data['Method'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, leakage_data['Overlapping_Patients']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('train_test_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(df, train_df, test_df, proper_train_df, proper_test_df, overlap):
    """Create a comprehensive summary report"""
    print(f"\nCreating Summary Report...")
    
    report_file = 'train_test_split_analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAIN-TEST SPLIT ANALYSIS REPORT\n")
        f.write("Top1 Action Class Depression Prediction\n")
        f.write("="*80 + "\n\n")
        
        # Dataset overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-"*30 + "\n")
        f.write(f"Total videos: {len(df)}\n")
        f.write(f"Unique patients: {df['Patient_ID'].nunique()}\n")
        f.write(f"Videos per patient (avg): {len(df) / df['Patient_ID'].nunique():.2f}\n")
        
        videos_per_patient = df.groupby('Patient_ID').size()
        f.write(f"Patients with multiple videos: {(videos_per_patient > 1).sum()}\n\n")
        
        # Current method analysis
        f.write("2. CURRENT SPLIT METHOD (Video-level)\n")
        f.write("-"*30 + "\n")
        f.write(f"Train set: {len(train_df)} videos from {train_df['Patient_ID'].nunique()} patients\n")
        f.write(f"Test set: {len(test_df)} videos from {test_df['Patient_ID'].nunique()} patients\n")
        f.write(f"Overlapping patients: {len(overlap)}\n")
        
        if len(overlap) > 0:
            f.write(f"⚠️  DATA LEAKAGE DETECTED: {len(overlap)} patients in both sets\n")
            f.write("This invalidates the test results!\n")
        else:
            f.write("✅ No data leakage detected\n")
        
        # Target distribution
        train_binary = train_df['Depression_Binary'].value_counts()
        test_binary = test_df['Depression_Binary'].value_counts()
        
        f.write(f"\nTarget Distribution (Binary):\n")
        f.write(f"Train - Not Depressed: {train_binary[0]} ({train_binary[0]/len(train_df)*100:.1f}%)\n")
        f.write(f"Train - Depressed: {train_binary[1]} ({train_binary[1]/len(train_df)*100:.1f}%)\n")
        f.write(f"Test - Not Depressed: {test_binary[0]} ({test_binary[0]/len(test_df)*100:.1f}%)\n")
        f.write(f"Test - Depressed: {test_binary[1]} ({test_binary[1]/len(test_df)*100:.1f}%)\n")
        
        train_balance = train_binary.min() / train_binary.max()
        test_balance = test_binary.min() / test_binary.max()
        f.write(f"Train balance ratio: {train_balance:.3f}\n")
        f.write(f"Test balance ratio: {test_balance:.3f}\n\n")
        
        # Recommended method
        f.write("3. RECOMMENDED SPLIT METHOD (Patient-level)\n")
        f.write("-"*30 + "\n")
        f.write(f"Train set: {len(proper_train_df)} videos from {proper_train_df['Patient_ID'].nunique()} patients\n")
        f.write(f"Test set: {len(proper_test_df)} videos from {proper_test_df['Patient_ID'].nunique()} patients\n")
        f.write("✅ No overlapping patients (proper separation)\n")
        
        proper_train_binary = proper_train_df['Depression_Binary'].value_counts()
        proper_test_binary = proper_test_df['Depression_Binary'].value_counts()
        
        f.write(f"\nTarget Distribution (Binary):\n")
        f.write(f"Train - Not Depressed: {proper_train_binary[0]} ({proper_train_binary[0]/len(proper_train_df)*100:.1f}%)\n")
        f.write(f"Train - Depressed: {proper_train_binary[1]} ({proper_train_binary[1]/len(proper_train_df)*100:.1f}%)\n")
        f.write(f"Test - Not Depressed: {proper_test_binary[0]} ({proper_test_binary[0]/len(proper_test_df)*100:.1f}%)\n")
        f.write(f"Test - Depressed: {proper_test_binary[1]} ({proper_test_binary[1]/len(proper_test_df)*100:.1f}%)\n")
        
        proper_train_balance = proper_train_binary.min() / proper_train_binary.max()
        proper_test_balance = proper_test_binary.min() / proper_test_binary.max()
        f.write(f"Train balance ratio: {proper_train_balance:.3f}\n")
        f.write(f"Test balance ratio: {proper_test_balance:.3f}\n\n")
        
        # Recommendations
        f.write("4. RECOMMENDATIONS\n")
        f.write("-"*30 + "\n")
        
        if len(overlap) > 0:
            f.write("🚨 CRITICAL: Current results are INVALID due to data leakage!\n")
            f.write("REQUIRED ACTIONS:\n")
            f.write("1. Re-implement patient-level splitting\n")
            f.write("2. Re-train all models using proper split\n")
            f.write("3. Re-evaluate model performance\n")
            f.write("4. Update comparison analysis\n\n")
        
        f.write("IMPLEMENTATION GUIDELINES:\n")
        f.write("1. Use patient-level stratification for train-test split\n")
        f.write("2. Ensure no patient appears in both train and test sets\n")
        f.write("3. Maintain target distribution balance across splits\n")
        f.write("4. Consider using cross-validation with patient-level folds\n\n")
        
        f.write("IMPACT ON RESULTS:\n")
        if len(overlap) > 0:
            f.write("- Current performance metrics are inflated (data leakage)\n")
            f.write("- True generalization performance is likely lower\n")
            f.write("- Model comparison results need revision\n")
        else:
            f.write("- Current results are valid (no data leakage)\n")
            f.write("- Performance metrics reflect true generalization\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved to: {report_file}")
    return report_file

def main():
    """Run the complete train-test split analysis"""
    print("Starting Train-Test Split Analysis")
    print("="*60)
    
    # Load and analyze dataset
    df = load_and_analyze_dataset()
    
    # Analyze current split method
    train_df, test_df, overlap = analyze_current_split_method(df)
    
    # Analyze target balance
    analyze_target_balance(train_df, test_df)
    
    # Suggest proper split method
    proper_train_df, proper_test_df = suggest_proper_split_method(df)
    
    # Create visualizations
    create_visualization(df, train_df, test_df, proper_train_df, proper_test_df)
    
    # Create summary report
    report_file = create_summary_report(df, train_df, test_df, proper_train_df, proper_test_df, overlap)
    
    print(f"\n{'='*60}")
    print("Train-Test Split Analysis Completed!")
    print("Generated files:")
    print("  - train_test_split_analysis.png")
    print(f"  - {report_file}")
    
    if len(overlap) > 0:
        print(f"\n🚨 CRITICAL FINDING: Data leakage detected!")
        print(f"   {len(overlap)} patients appear in both train and test sets")
        print(f"   Current results are INVALID and need correction!")
    else:
        print(f"\n✅ No data leakage: Current split method is valid")
    
    print(f"{'='*60}")
    
    return df, train_df, test_df, proper_train_df, proper_test_df, overlap

if __name__ == "__main__":
    results = main()