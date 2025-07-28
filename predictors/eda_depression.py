"""
Exploratory Data Analysis (EDA) for Depression Prediction Dataset
This script performs comprehensive EDA and data type conversions for the depression dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from scipy.stats import chi2_contingency
import os

warnings.filterwarnings('ignore')

class DepressionEDA:
    def __init__(self, dataset_path='../processed_data/depression_processed.csv'):
        """Initialize the EDA class with dataset path"""
        self.dataset_path = dataset_path
        self.df = None
        self.feature_cols = None
        self.target_cols = None
        self.metadata_cols = None
        
    def load_data(self):
        """Load and perform initial data inspection"""
        print("🔍 Loading Depression Dataset...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Identify column types
        self.feature_cols = [col for col in self.df.columns if col.startswith('cluster_')]
        self.target_cols = ['Depression_Binary', 'Depression_3Class', 'Binary_Depression']
        self.metadata_cols = ['Patient_ID']  # Simplified for new dataset structure
        
        print(f"\n📊 Column Analysis:")
        print(f"Feature columns (clusters): {len(self.feature_cols)}")
        print(f"Target columns: {len(self.target_cols)}")
        print(f"Metadata columns: {len(self.metadata_cols)}")
        
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("📋 BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"\nDataset Info:")
        print(self.df.info())
        
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        
        print(f"\nMissing Values:")
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
    def target_analysis(self):
        """Analyze target variables"""
        print("\n" + "="*60)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        # Binary Depression Analysis
        print(f"\n1. Depression_Binary Distribution:")
        binary_counts = self.df['Depression_Binary'].value_counts()
        binary_percent = self.df['Depression_Binary'].value_counts(normalize=True) * 100
        
        print(f"Class 0 (Not Depressed): {binary_counts[0]} ({binary_percent[0]:.1f}%)")
        print(f"Class 1 (Depressed): {binary_counts[1]} ({binary_percent[1]:.1f}%)")
        
        # 3-Class Depression Analysis
        print(f"\n2. Depression_3Class Distribution:")
        class3_counts = self.df['Depression_3Class'].value_counts().sort_index()
        class3_percent = self.df['Depression_3Class'].value_counts(normalize=True).sort_index() * 100
        
        for cls in class3_counts.index:
            print(f"Class {cls}: {class3_counts[cls]} ({class3_percent[cls]:.1f}%)")
        
        # Text Binary Depression Analysis
        print(f"\n3. Binary_Depression (Text) Distribution:")
        text_counts = self.df['Binary_Depression'].value_counts()
        text_percent = self.df['Binary_Depression'].value_counts(normalize=True) * 100
        
        for label in text_counts.index:
            print(f"{label}: {text_counts[label]} ({text_percent[label]:.1f}%)")
            
        # Check target consistency
        print(f"\n4. Target Variable Consistency Check:")
        consistency_check = pd.crosstab(self.df['Depression_Binary'], self.df['Binary_Depression'])
        print(consistency_check)
        
    def feature_analysis(self):
        """Analyze cluster features"""
        print("\n" + "="*60)
        print("🔬 FEATURE ANALYSIS")
        print("="*60)
        
        # Basic statistics for cluster features
        cluster_data = self.df[self.feature_cols]
        
        print(f"\nCluster Features Statistics:")
        print(f"Number of features: {len(self.feature_cols)}")
        print(f"Feature value range: [{cluster_data.min().min():.6f}, {cluster_data.max().max():.6f}]")
        print(f"Mean of all features: {cluster_data.mean().mean():.6f}")
        print(f"Std of all features: {cluster_data.std().mean():.6f}")
        
        # Check for zero variance features
        zero_variance_features = cluster_data.columns[cluster_data.var() == 0].tolist()
        print(f"\nZero variance features: {len(zero_variance_features)}")
        if zero_variance_features:
            print(f"Features with zero variance: {zero_variance_features[:10]}...")  # Show first 10
            
        # Check for highly correlated features
        correlation_matrix = cluster_data.corr()
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        correlation_matrix.iloc[i, j]
                    ))
        
        print(f"\nHighly correlated feature pairs (>0.9): {len(high_corr_pairs)}")
        if high_corr_pairs:
            for pair in high_corr_pairs[:5]:  # Show first 5
                print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
                
    def metadata_analysis(self):
        """Analyze metadata columns"""
        print("\n" + "="*60)
        print("📊 METADATA ANALYSIS")
        print("="*60)
        
        # PHQ9 Score Analysis
        if 'PHQ9_Score' in self.df.columns:
            print(f"\nPHQ9 Score Analysis:")
            phq9_stats = self.df['PHQ9_Score'].describe()
            print(phq9_stats)
            
            # PHQ9 by depression status
            print(f"\nPHQ9 Score by Depression Status:")
            phq9_by_depression = self.df.groupby('Binary_Depression')['PHQ9_Score'].agg(['mean', 'std', 'count'])
            print(phq9_by_depression)
            
        # HRSD Score Analysis
        if 'HRSD_Score' in self.df.columns:
            print(f"\nHRSD Score Analysis:")
            hrsd_stats = self.df['HRSD_Score'].describe()
            print(hrsd_stats)
            
        # ADS Score Analysis
        if 'ADS_Score' in self.df.columns:
            print(f"\nADS Score Analysis:")
            ads_stats = self.df['ADS_Score'].describe()
            print(ads_stats)
            
        # Confidence Analysis
        if 'Confidence' in self.df.columns:
            print(f"\nConfidence Level Distribution:")
            confidence_counts = self.df['Confidence'].value_counts()
            print(confidence_counts)
            
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        # Create output directory for plots
        os.makedirs('visualizations', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Target Distribution Plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Binary depression distribution
        self.df['Depression_Binary'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Depression Binary Distribution')
        axes[0,0].set_xlabel('Depression Status (0=No, 1=Yes)')
        axes[0,0].set_ylabel('Count')
        
        # 3-Class depression distribution
        self.df['Depression_3Class'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Depression 3-Class Distribution')
        axes[0,1].set_xlabel('Depression Class')
        axes[0,1].set_ylabel('Count')
        
        # Text binary depression distribution
        self.df['Binary_Depression'].value_counts().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Binary Depression (Text) Distribution')
        axes[1,0].set_xlabel('Depression Status')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Overall depression status
        if 'Overall_Depression_Status' in self.df.columns:
            self.df['Overall_Depression_Status'].value_counts().plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Overall Depression Status Distribution')
            axes[1,1].set_xlabel('Depression Status')
            axes[1,1].set_ylabel('Count')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/target_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature means distribution
        feature_means = self.df[self.feature_cols].mean()
        axes[0,0].hist(feature_means, bins=30, alpha=0.7)
        axes[0,0].set_title('Distribution of Feature Means')
        axes[0,0].set_xlabel('Mean Value')
        axes[0,0].set_ylabel('Frequency')
        
        # Feature standard deviations
        feature_stds = self.df[self.feature_cols].std()
        axes[0,1].hist(feature_stds, bins=30, alpha=0.7)
        axes[0,1].set_title('Distribution of Feature Standard Deviations')
        axes[0,1].set_xlabel('Standard Deviation')
        axes[0,1].set_ylabel('Frequency')
        
        # Sample feature distributions
        sample_features = self.feature_cols[:4]  # First 4 features
        for i, feature in enumerate(sample_features):
            row = (i // 2) if i < 2 else 1
            col = i % 2 if i < 2 else (i - 2)
            if i < 2:
                self.df[feature].hist(bins=30, alpha=0.7, ax=axes[row, col])
                axes[row, col].set_title(f'Distribution of {feature}')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation Heatmap (sample)
        plt.figure(figsize=(12, 10))
        sample_features = self.feature_cols[:20]  # First 20 features for visibility
        corr_matrix = self.df[sample_features].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (Sample Features)')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Score distributions by depression status
        if all(col in self.df.columns for col in ['PHQ9_Score', 'HRSD_Score', 'ADS_Score']):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # PHQ9 Score by depression
            self.df.boxplot(column='PHQ9_Score', by='Binary_Depression', ax=axes[0])
            axes[0].set_title('PHQ9 Score by Depression Status')
            axes[0].set_xlabel('Depression Status')
            axes[0].set_ylabel('PHQ9 Score')
            
            # HRSD Score by depression
            self.df.boxplot(column='HRSD_Score', by='Binary_Depression', ax=axes[1])
            axes[1].set_title('HRSD Score by Depression Status')
            axes[1].set_xlabel('Depression Status')
            axes[1].set_ylabel('HRSD Score')
            
            # ADS Score by depression
            self.df.boxplot(column='ADS_Score', by='Binary_Depression', ax=axes[2])
            axes[2].set_title('ADS Score by Depression Status')
            axes[2].set_xlabel('Depression Status')
            axes[2].set_ylabel('ADS Score')
            
            plt.tight_layout()
            plt.savefig('visualizations/scores_by_depression.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved to 'visualizations/' directory")
        
    def data_preprocessing(self):
        """Perform data type conversions and preprocessing"""
        print("\n" + "="*60)
        print("🔧 DATA PREPROCESSING & TYPE CONVERSIONS")
        print("="*60)
        
        # Create a copy for preprocessing
        processed_df = self.df.copy()
        
        # 1. Handle missing values
        print(f"\nHandling missing values...")
        missing_before = processed_df.isnull().sum().sum()
        
        # Fill missing numeric values with median
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                
        missing_after = processed_df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}, after: {missing_after}")
        
        # 2. Data type conversions
        print(f"\nPerforming data type conversions...")
        
        # Ensure numeric columns are proper numeric types
        for col in self.feature_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
        # Convert target variables to appropriate types
        processed_df['Depression_Binary'] = processed_df['Depression_Binary'].astype('int8')
        processed_df['Depression_3Class'] = processed_df['Depression_3Class'].astype('int8')
        
        # Convert Patient_ID to string
        processed_df['Patient_ID'] = processed_df['Patient_ID'].astype(str)
        
        # Convert boolean columns
        if 'SKID_Depressed' in processed_df.columns:
            processed_df['SKID_Depressed'] = processed_df['SKID_Depressed'].astype(bool)
        
        # 3. Feature scaling (StandardScaler)
        print(f"\nApplying feature scaling...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_df[self.feature_cols])
        
        # Create scaled feature dataframe
        scaled_feature_df = pd.DataFrame(
            scaled_features, 
            columns=[f"{col}_scaled" for col in self.feature_cols],
            index=processed_df.index
        )
        
        # Combine with original data
        final_df = pd.concat([
            processed_df[['Patient_ID'] + self.target_cols + self.metadata_cols],
            processed_df[self.feature_cols],  # Original features
            scaled_feature_df  # Scaled features
        ], axis=1)
        
        # 4. Create additional engineered features
        print(f"\nCreating engineered features...")
        
        # Total cluster activity
        final_df['total_cluster_activity'] = processed_df[self.feature_cols].sum(axis=1)
        
        # Most active cluster
        final_df['most_active_cluster'] = processed_df[self.feature_cols].idxmax(axis=1)
        
        # Number of active clusters (> 0)
        final_df['num_active_clusters'] = (processed_df[self.feature_cols] > 0).sum(axis=1)
        
        # Cluster diversity (entropy-like measure)
        cluster_data = processed_df[self.feature_cols]
        final_df['cluster_diversity'] = -np.sum(
            cluster_data * np.log(cluster_data + 1e-10), axis=1
        )
        
        print(f"✅ Preprocessing complete!")
        print(f"Final dataset shape: {final_df.shape}")
        print(f"New features added: 4 engineered features + {len(self.feature_cols)} scaled features")
        
        return final_df, scaler
    
    def save_processed_data(self, processed_df, scaler):
        """Save processed data and preprocessing objects"""
        print(f"\n💾 Saving processed data...")
        
        # Create output directory
        os.makedirs('processed_data', exist_ok=True)
        
        # Save processed dataset
        processed_df.to_csv('processed_data/depression_processed.csv', index=False)
        
        # Save feature and target information
        feature_info = {
            'original_features': self.feature_cols,
            'scaled_features': [f"{col}_scaled" for col in self.feature_cols],
            'target_columns': self.target_cols,
            'metadata_columns': self.metadata_cols,
            'engineered_features': ['total_cluster_activity', 'most_active_cluster', 
                                   'num_active_clusters', 'cluster_diversity']
        }
        
        # Save as pickle for easy loading
        import pickle
        with open('processed_data/feature_info.pkl', 'wb') as f:
            pickle.dump(feature_info, f)
            
        with open('processed_data/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"✅ Processed data saved to 'processed_data/' directory")
        print(f"  - depression_processed.csv: Main processed dataset")
        print(f"  - feature_info.pkl: Feature column information")
        print(f"  - scaler.pkl: Fitted StandardScaler object")
        
        return feature_info
    
    def run_complete_eda(self):
        """Run the complete EDA pipeline"""
        print("🚀 Starting Complete EDA Pipeline for Depression Dataset")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Basic information
        self.basic_info()
        
        # Target analysis
        self.target_analysis()
        
        # Feature analysis
        self.feature_analysis()
        
        # Metadata analysis
        self.metadata_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Data preprocessing
        processed_df, scaler = self.data_preprocessing()
        
        # Save processed data
        feature_info = self.save_processed_data(processed_df, scaler)
        
        print("\n🎉 EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Dataset: {self.df.shape[0]} samples, {len(self.feature_cols)} cluster features")
        print(f"📈 Target: Depression prediction (Binary & 3-Class)")
        print(f"💾 Processed data saved for ML training")
        print(f"📊 Visualizations created in 'visualizations/' directory")
        
        return processed_df, scaler, feature_info

def main():
    """Main function to run EDA"""
    # Initialize EDA class
    eda = DepressionEDA()
    
    # Run complete EDA
    processed_df, scaler, feature_info = eda.run_complete_eda()
    
    return processed_df, scaler, feature_info

if __name__ == "__main__":
    processed_data, scaler, feature_info = main() 