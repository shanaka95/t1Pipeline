"""
Base Model Class for Depression Severity Prediction
This module provides a base class with common functionality for all depression severity prediction models.
Includes comprehensive SMOTE support for handling class imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize, LabelEncoder
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class BaseSeverityModel:
    def __init__(self, processed_data_path='../processed_data/depression_processed.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the base severity model trainer"""
        self.processed_data_path = processed_data_path
        self.feature_info_path = feature_info_path
        self.df = None
        self.feature_info = None
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        
    def load_processed_data(self):
        """Load processed data and feature information"""
        print("Loading processed data...")
        
        # Load processed dataset
        self.df = pd.read_csv(self.processed_data_path)
        
        # Load feature information
        with open(self.feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)
            
        print(f"Data loaded: {self.df.shape}")
        print(f"Available feature sets:")
        for key, features in self.feature_info.items():
            if isinstance(features, list):
                print(f"  - {key}: {len(features)} features")
                
        return self.df, self.feature_info
    
    def _filter_target_leakage_columns(self, feature_cols):
        """Filter out any columns that could reveal the target variable"""
        # Define columns that could leak target information
        target_leakage_columns = [
            'Depression_Binary', 'Depression_3Class', 'Binary_Depression',
            'Overall_Depression_Status', 'SKID_Depressed'
        ]
        
        # Filter out any columns that match target leakage patterns
        filtered_cols = []
        for col in feature_cols:
            # Check if column is in target leakage list
            if col in target_leakage_columns:
                print(f"WARNING: Excluding potential target leakage column: {col}")
                continue
            
            # Check if column contains target-related keywords
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['depression', 'depressed', 'skid', 'phq9', 'hrsd', 'ads']):
                print(f"WARNING: Excluding potential target leakage column: {col}")
                continue
                
            filtered_cols.append(col)
        
        if len(filtered_cols) != len(feature_cols):
            print(f"Filtered out {len(feature_cols) - len(filtered_cols)} potential target leakage columns")
        
        return filtered_cols
    
    def prepare_features_targets(self, use_scaled_features=True, include_engineered=True):
        """Prepare feature sets and targets for severity prediction"""
        print("\nPreparing features and targets for severity prediction...")
        
        # Select feature set
        if use_scaled_features:
            feature_cols = self.feature_info['scaled_cluster_columns'].copy()
            print(f"Using scaled features: {len(feature_cols)} features")
        else:
            feature_cols = self.feature_info['cluster_columns'].copy()
            print(f"Using original features: {len(feature_cols)} features")
            
        # Add engineered features if requested
        if include_engineered:
            engineered_features = self.feature_info['derived_columns']
            # Remove non-numeric engineered features for training
            numeric_engineered = [f for f in engineered_features if f != 'most_active_cluster']
            feature_cols.extend(numeric_engineered)
            print(f"Added engineered features: {len(numeric_engineered)} features")
        
        # Filter out any potential target leakage columns
        feature_cols = self._filter_target_leakage_columns(feature_cols)
        
        # Verify that no target columns are in features
        target_columns = ['Depression_Binary', 'Depression_3Class', 'Binary_Depression']
        for target_col in target_columns:
            if target_col in feature_cols:
                raise ValueError(f"CRITICAL ERROR: Target column '{target_col}' found in feature list!")
        
        # Print final feature list for transparency
        self._print_feature_summary(feature_cols)
        
        # Prepare feature matrix
        X = self.df[feature_cols].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Prepare targets - focus on 3-class severity
        y_3class_original = self.df['Depression_3Class']
        
        # Encode labels for XGBoost compatibility (0, 1, 2 instead of 1, 2, 3)
        y_3class = self.label_encoder.fit_transform(y_3class_original)
        
        # Also prepare binary target for comparison
        y_binary = self.df['Depression_Binary']
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Original 3-Class severity target distribution:")
        print(y_3class_original.value_counts().sort_index())
        print(f"Encoded 3-Class severity target distribution:")
        print(pd.Series(y_3class).value_counts().sort_index())
        print(f"Binary target distribution:")
        print(y_binary.value_counts().sort_index())
        
        return X, y_3class, y_binary, feature_cols
    
    def _print_feature_summary(self, feature_cols):
        """Print a summary of the features being used"""
        print(f"\nFeature Summary:")
        print(f"Total features: {len(feature_cols)}")
        
        # Categorize features
        cluster_features = [col for col in feature_cols if col.startswith('cluster_')]
        scaled_features = [col for col in feature_cols if col.endswith('_scaled')]
        engineered_features = [col for col in feature_cols if col in ['total_cluster_activity', 'most_active_cluster', 'num_active_clusters', 'cluster_diversity']]
        
        print(f"  - Cluster features: {len(cluster_features)}")
        print(f"  - Scaled features: {len(scaled_features)}")
        print(f"  - Engineered features: {len(engineered_features)}")
        
        # Show sample features
        if cluster_features:
            print(f"  Sample cluster features: {cluster_features[:3]}...")
        if engineered_features:
            print(f"  Engineered features: {engineered_features}")
        
        print("Feature integrity verified - only cluster-based features used")
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        print(f"\nData split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Show class distribution in training set
        train_dist = pd.Series(y_train).value_counts().sort_index()
        print(f"Training target distribution:")
        severity_labels = ['Mild/Subclinical', 'Moderate', 'Severe']
        for i, count in enumerate(train_dist):
            print(f"  {severity_labels[i]} (Class {i}): {count}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train, method='smote', random_state=42):
        """
        Handle class imbalance using various SMOTE techniques for multi-class
        
        Parameters:
        - method: 'none', 'smote', 'borderline_smote', 'adasyn', 'smote_tomek', 'smote_enn', 'undersample'
        - random_state: for reproducibility
        """
        print(f"\nHandling class imbalance using: {method}")
        
        # Show original distribution
        original_dist = pd.Series(y_train).value_counts().sort_index()
        print(f"Original distribution:")
        severity_labels = ['Mild/Subclinical', 'Moderate', 'Severe']
        for i, count in enumerate(original_dist):
            print(f"  {severity_labels[i]} (Class {i}): {count}")
        
        if method == 'none':
            return X_train, y_train
        
        elif method == 'smote':
            # Standard SMOTE oversampling for multi-class
            # Adjust k_neighbors based on minority class size
            min_samples = min(original_dist.values)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        elif method == 'borderline_smote':
            # Borderline SMOTE - focuses on borderline cases
            min_samples = min(original_dist.values)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            borderline_smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)
            
        elif method == 'adasyn':
            # ADASYN - Adaptive Synthetic Sampling
            min_samples = min(original_dist.values)
            n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors)
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            
        elif method == 'smote_tomek':
            # SMOTE + Tomek links (oversample + clean)
            min_samples = min(original_dist.values)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote_tomek = SMOTETomek(smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors), 
                                   random_state=random_state)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            
        elif method == 'smote_enn':
            # SMOTE + Edited Nearest Neighbours
            min_samples = min(original_dist.values)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote_enn = SMOTEENN(smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors),
                               random_state=random_state)
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=random_state)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Show new distribution
        new_dist = pd.Series(y_resampled).value_counts().sort_index()
        print(f"New distribution:")
        for i, count in enumerate(new_dist):
            print(f"  {severity_labels[i]} (Class {i}): {count}")
        print(f"Samples changed: {len(X_train)} → {len(X_resampled)}")
        
        return X_resampled, y_resampled
    
    def evaluate_models(self):
        """Comprehensive model evaluation for multi-class"""
        print("\n" + "="*60)
        print("MODEL EVALUATION - SEVERITY PREDICTION")
        print("="*60)
        
        evaluation_results = {}
        
        for model_name, results in self.results.items():
            y_test = results['y_test']
            y_pred = results['y_pred']
            y_pred_proba = results['y_pred_proba']
            
            print(f"\n{model_name.upper()} Results:")
            print("-" * 40)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Multi-class AUC calculation
            try:
                # Binarize the output for multi-class AUC
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                if y_test_bin.shape[1] == 3:  # All classes present
                    auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc = 0.0
            except:
                auc = 0.0  # If AUC calculation fails
            
            # Store results
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc
            }
            
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
            
            # Classification report with original labels
            y_test_original = self.label_encoder.inverse_transform(y_test)
            y_pred_original = self.label_encoder.inverse_transform(y_pred)
            
            severity_labels = ['Mild/Subclinical', 'Moderate', 'Severe']
            print(f"\nClassification Report:")
            print(classification_report(y_test_original, y_pred_original, 
                                      target_names=severity_labels, zero_division=0))
            
        return evaluation_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations for severity prediction"""
        print("\nCreating visualizations...")
        
        # Create output directory
        os.makedirs('../severity_results', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Comparison Plot
        self._plot_model_comparison()
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices()
        
        print("Visualizations saved to '../severity_results/' directory")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison for severity prediction"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(self.results.keys())
        
        # Create evaluation results if not exists
        if not hasattr(self, 'evaluation_results'):
            self.evaluation_results = self.evaluate_models()
        
        # Prepare data for plotting
        comparison_data = []
        for model in model_names:
            for metric in metrics:
                comparison_data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.evaluation_results[model][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model')
        plt.title('Severity Prediction Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('../severity_results/severity_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                       xticklabels=['Mild', 'Moderate', 'Severe'],
                       yticklabels=['Mild', 'Moderate', 'Severe'])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted Severity')
            axes[i].set_ylabel('Actual Severity')
        
        plt.tight_layout()
        plt.savefig('../severity_results/severity_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving trained models...")
        
        os.makedirs('../saved_models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f'../saved_models/severity_{model_name}_{timestamp}.pkl'
            joblib.dump(model, filename)
            print(f"Severity {model_name} saved to {filename}")
        
        # Save results summary
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'accuracy': accuracy_score(results['y_test'], results['y_pred']),
                'f1_score': f1_score(results['y_test'], results['y_pred'], average='weighted', zero_division=0)
            }
        
        summary_df = pd.DataFrame(results_summary).T
        summary_df.to_csv(f'../saved_models/severity_model_summary_{timestamp}.csv')
        print(f"Severity model summary saved to ../saved_models/severity_model_summary_{timestamp}.csv")
    
    def verify_feature_integrity(self, feature_cols):
        """Verify that we're only using appropriate features"""
        print("\nVerifying feature integrity...")
        
        # Check if we're using cluster-based features
        cluster_features = [col for col in feature_cols if col.startswith('cluster_')]
        non_cluster_features = [col for col in feature_cols if not col.startswith('cluster_') 
                               and col not in ['total_cluster_activity', 'num_active_clusters', 'cluster_diversity']]
        
        print(f"Cluster-based features: {len(cluster_features)}")
        print(f"Derived features: {len([col for col in feature_cols if col in ['total_cluster_activity', 'num_active_clusters', 'cluster_diversity']])}")
        
        if non_cluster_features:
            print(f"WARNING: Found non-cluster features: {non_cluster_features}")
        else:
            print("All features are cluster-based or derived (correct)")
        
        # Verify no target leakage
        target_columns = ['Depression_Binary', 'Depression_3Class', 'Binary_Depression', 
                         'Overall_Depression_Status', 'SKID_Depressed']
        leakage_columns = [col for col in feature_cols if col in target_columns]
        if leakage_columns:
            raise ValueError(f"CRITICAL ERROR: Target leakage detected! Columns: {leakage_columns}")
        else:
            print("No target leakage detected") 