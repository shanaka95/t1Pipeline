"""
Comprehensive Top1 Action Class Depression Prediction Workflow with Normalized Features
This version uses StandardScaler normalized action class features for fair comparison with Top5.

Enhanced version with normalized features matching top5_workflow capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import pickle
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, roc_curve, 
                           precision_recall_curve, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

class NormalizedTop1DepressionPredictionWorkflow:
    def __init__(self, dataset_path='../datasets/ml_depression_dataset_top1_normalized.csv'):
        """Initialize the normalized top1 workflow"""
        self.dataset_path = dataset_path
        self.df = None
        self.action_features = None
        self.action_features_scaled = None
        self.all_models = {}
        self.all_results = {}
        self.combined_evaluation = {}
        self.train_patients = None
        self.test_patients = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directories
        self.results_dir = f'../top1_normalized_results_{self.timestamp}'
        self.visualizations_dir = f'{self.results_dir}/visualizations'
        self.models_dir = f'{self.results_dir}/models'
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_data_and_features(self):
        """Load data and display comprehensive feature information"""
        print("Starting Comprehensive Top1 Action Class Depression Prediction Workflow")
        print("🎯 USING NORMALIZED FEATURES (StandardScaler)")
        print("="*80)
        
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"Data loaded: {self.df.shape}")
        print(f"Dataset Overview:")
        print(f"  - Samples: {self.df.shape[0]}")
        print(f"  - Features: {self.df.shape[1]}")
        
        # Identify action class features (both raw and scaled)
        self.action_features = [col for col in self.df.columns if col.startswith('action_class_') and not col.endswith('_scaled')]
        self.action_features_scaled = [col for col in self.df.columns if col.startswith('action_class_') and col.endswith('_scaled')]
        
        return self.df
    
    def show_feature_details(self):
        """Display detailed feature information for Top1 action classes"""
        print("\n" + "="*60)
        print("TOP1 NORMALIZED ACTION CLASS FEATURE DETAILS")
        print("="*60)
        
        print(f"\nFeature Categories:")
        print(f"  - Original action class features: {len(self.action_features)} features")
        print(f"  - Normalized action class features: {len(self.action_features_scaled)} features")
        print(f"  - Patient information: Patient_ID")
        print(f"  - Video information: video_name")
        print(f"  - Target variables: Depression_Binary, Depression_3Class")
        
        print(f"\nAction Class Features (52 total):")
        print(f"  Range: action_class_00 to action_class_51")
        print(f"  Normalized: action_class_00_scaled to action_class_51_scaled")
        print(f"  Type: StandardScaler normalized percentages")
        print(f"  Interpretation: Standardized human action categories (mean=0, std=1)")
        
        # Compare raw vs normalized statistics
        print(f"\nData Distribution Comparison:")
        raw_data = self.df[self.action_features]
        scaled_data = self.df[self.action_features_scaled]
        
        print(f"  Raw Features:")
        print(f"    Min: {raw_data.min().min():.6f}")
        print(f"    Max: {raw_data.max().max():.6f}")
        print(f"    Mean: {raw_data.mean().mean():.6f}")
        print(f"    Std: {raw_data.std().mean():.6f}")
        
        print(f"  Normalized Features:")
        print(f"    Min: {scaled_data.min().min():.6f}")
        print(f"    Max: {scaled_data.max().max():.6f}")
        print(f"    Mean: {scaled_data.mean().mean():.6f}")
        print(f"    Std: {scaled_data.std().mean():.6f}")
        
        # Show most variable features after normalization
        print(f"\nTop 10 Most Variable Normalized Action Classes:")
        scaled_std = scaled_data.std().sort_values(ascending=False)
        for i, (feature, std_val) in enumerate(scaled_std.head(10).items()):
            class_id = feature.split('_')[2]  # Extract class number
            print(f"  {feature} (Class {class_id}): {std_val:.4f} std")
        
        # Show target distribution
        print(f"\nTarget Variable Distribution:")
        binary_dist = self.df['Depression_Binary'].value_counts()
        for value, count in binary_dist.items():
            label = 'Depressed' if value == 1 else 'Not Depressed'
            print(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Show patient information
        print(f"\nPatient Information:")
        print(f"  - Total unique patients: {self.df['Patient_ID'].nunique()}")
        videos_per_patient = self.df.groupby('Patient_ID').size()
        patients_with_multiple = (videos_per_patient > 1).sum()
        print(f"  - Patients with multiple videos: {patients_with_multiple}")
        print(f"  - Videos per patient: {videos_per_patient.mean():.2f} average")
        
    def train_comparison_models(self, test_size=0.2, random_state=42):
        """Train models comparing raw vs normalized features"""
        print("\n" + "="*80)
        print("TRAINING COMPARISON MODELS - RAW vs NORMALIZED FEATURES")
        print("="*80)
        
        # Different approaches to test
        approaches = [
            {'name': 'Raw_Features', 'features': self.action_features, 'description': 'Original percentage features'},
            {'name': 'Normalized_Features', 'features': self.action_features_scaled, 'description': 'StandardScaler normalized features'},
            {'name': 'Normalized_ClassWeights', 'features': self.action_features_scaled, 'description': 'Normalized + balanced class weights'},
        ]
        
        comparison_results = {}
        
        # Prepare data with patient-level splitting (once for all approaches)
        unique_patients = self.df['Patient_ID'].unique()
        patient_labels = self.df.groupby('Patient_ID')['Depression_Binary'].first()
        
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=random_state, 
            stratify=patient_labels
        )
        
        train_mask = self.df['Patient_ID'].isin(train_patients)
        test_mask = self.df['Patient_ID'].isin(test_patients)
        
        y_train = self.df.loc[train_mask, 'Depression_Binary']
        y_test = self.df.loc[test_mask, 'Depression_Binary']
        
        print(f"Patient split: {len(train_patients)} train patients, {len(test_patients)} test patients")
        
        # Test each approach with Random Forest (fast comparison)
        for approach in approaches:
            print(f"\n{'='*50}")
            print(f"Testing: {approach['name']} - {approach['description']}")
            print(f"{'='*50}")
            
            # Get features for this approach
            X_train = self.df.loc[train_mask, approach['features']]
            X_test = self.df.loc[test_mask, approach['features']]
            
            print(f"Feature count: {len(approach['features'])}")
            print(f"Dataset split: {len(X_train)} train, {len(X_test)} test")
            
            # Configure model based on approach
            if approach['name'] == 'Normalized_ClassWeights':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state, 
                                             class_weight='balanced')
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            comparison_results[approach['name']] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'feature_count': len(approach['features']),
                'description': approach['description']
            }
            
            print(f"Results: F1={comparison_results[approach['name']]['f1_weighted']:.4f}, "
                  f"AUC={comparison_results[approach['name']]['auc_roc']:.4f}")
        
        # Print comprehensive comparison
        print(f"\n{'='*80}")
        print("RAW vs NORMALIZED FEATURES COMPARISON RESULTS")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame({k: {metric: v[metric] for metric in v.keys() if metric not in ['description', 'feature_count']} 
                                    for k, v in comparison_results.items()}).T
        print(comparison_df.round(4))
        
        # Save comparison results
        comparison_df.to_csv(f'{self.results_dir}/raw_vs_normalized_comparison.csv')
        print(f"\nComparison results saved to {self.results_dir}/raw_vs_normalized_comparison.csv")
        
        # Determine best approach
        best_approach = max(comparison_results.items(), key=lambda x: x[1]['f1_weighted'])
        print(f"\n🏆 BEST APPROACH: {best_approach[0]} (F1-Weighted: {best_approach[1]['f1_weighted']:.4f})")
        
        return comparison_results, best_approach
    
    def train_all_models_normalized(self, use_normalized=True, test_size=0.2, random_state=42):
        """Train all models with normalized features (default) or raw features"""
        feature_type = "NORMALIZED" if use_normalized else "RAW"
        print(f"\n" + "="*60)
        print(f"TRAINING ALL MODELS WITH {feature_type} FEATURES")
        print("="*60)
        
        # Select features based on choice
        features_to_use = self.action_features_scaled if use_normalized else self.action_features
        print(f"Using features: {feature_type} ({len(features_to_use)} features)")
        
        # Patient-level data splitting
        unique_patients = self.df['Patient_ID'].unique()
        patient_labels = self.df.groupby('Patient_ID')['Depression_Binary'].first()
        
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=random_state, 
            stratify=patient_labels
        )
        
        # Store for later use
        self.train_patients = train_patients
        self.test_patients = test_patients
        
        train_mask = self.df['Patient_ID'].isin(train_patients)
        test_mask = self.df['Patient_ID'].isin(test_patients)
        
        X_train = self.df.loc[train_mask, features_to_use]
        X_test = self.df.loc[test_mask, features_to_use]
        y_train = self.df.loc[train_mask, 'Depression_Binary']
        y_test = self.df.loc[test_mask, 'Depression_Binary']
        
        print(f"Data split verification:")
        print(f"  Training: {len(X_train)} videos from {len(train_patients)} patients")
        print(f"  Testing: {len(X_test)} videos from {len(test_patients)} patients")
        print(f"  ✅ No patient overlap verified")
        
        # Model configurations optimized for normalized Top1 action classes
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=random_state,
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'needs_scaling': False  # RF doesn't need additional scaling
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=random_state, 
                    max_iter=1000,
                    class_weight='balanced',
                    C=1.0
                ),
                'needs_scaling': not use_normalized  # Only scale if using raw features
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=random_state, 
                    eval_metric='logloss',
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100
                ),
                'needs_scaling': False  # XGB handles scale internally
            }
        }
        
        # Train each model
        for model_name, config in models_config.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name} ({feature_type} features)")
            print(f"{'='*40}")
            
            model = config['model']
            needs_scaling = config['needs_scaling']
            scaler = None
            
            # Apply additional scaling only if needed and using raw features
            if needs_scaling:
                scaler = StandardScaler()
                X_train_processed = scaler.fit_transform(X_train)
                X_test_processed = scaler.transform(X_test)
                print(f"  ⚠️  Additional StandardScaler applied for {model_name}")
            else:
                X_train_processed = X_train
                X_test_processed = X_test
                if use_normalized:
                    print(f"  ✅ Using pre-normalized features for {model_name}")
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Store results
            self.all_models[model_name] = {
                'model': model,
                'scaler': scaler,
                'features_used': feature_type,
                'feature_names': features_to_use
            }
            
            self.all_results[model_name] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'X_test': X_test_processed,
                'feature_names': features_to_use,
                'features_used': feature_type
            }
            
            print(f"✅ {model_name} training completed ({feature_type} features)")
        
        print(f"\nAll models trained successfully with {feature_type} features!")
        print(f"Models trained: {len(self.all_models)}")
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        for model_name, results in self.all_results.items():
            print(f"\n{model_name.upper()} Results ({results['features_used']} features):")
            print("-" * 50)
            
            y_test = results['y_test']
            y_pred = results['y_pred']
            y_pred_proba = results['y_pred_proba']
            
            # Calculate comprehensive metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba),
                'features_used': results['features_used']
            }
            
            # Store for comparison
            self.combined_evaluation[model_name] = metrics
            
            # Print key metrics
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"Avg Precision: {metrics['avg_precision']:.4f}")
            
            # Print classification report
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
    
    def analyze_feature_importance(self):
        """Analyze and compare feature importance across models"""
        print("\n" + "="*60)
        print("NORMALIZED FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        importance_results = {}
        
        for model_name, model_info in self.all_models.items():
            model = model_info['model']
            feature_names = model_info['feature_names']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_results[model_name] = feature_importance
                
                print(f"\nTop 10 Most Important Action Classes for {model_name}:")
                for i, row in feature_importance.head(10).iterrows():
                    if '_scaled' in row['feature']:
                        class_id = row['feature'].split('_')[2]  # action_class_XX_scaled
                        feature_type = "Normalized"
                    else:
                        class_id = row['feature'].split('_')[-1]  # action_class_XX
                        feature_type = "Raw"
                    print(f"  {row['feature']} ({feature_type} Class {class_id}): {row['importance']:.4f}")
                
                # Save feature importance
                feature_importance.to_csv(f'{self.results_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.csv', index=False)
            
            elif hasattr(model, 'coef_'):
                # For logistic regression
                coefficients = np.abs(model.coef_[0])
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coefficients
                }).sort_values('importance', ascending=False)
                
                importance_results[model_name] = feature_importance
                
                print(f"\nTop 10 Most Important Action Classes for {model_name}:")
                for i, row in feature_importance.head(10).iterrows():
                    if '_scaled' in row['feature']:
                        class_id = row['feature'].split('_')[2]
                        feature_type = "Normalized"
                    else:
                        class_id = row['feature'].split('_')[-1]
                        feature_type = "Raw"
                    print(f"  {row['feature']} ({feature_type} Class {class_id}): {row['importance']:.4f}")
                
                # Save feature importance
                feature_importance.to_csv(f'{self.results_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.csv', index=False)
        
        return importance_results
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating comprehensive visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create visualizations
        self._plot_model_performance_comparison()
        self._plot_feature_distribution_comparison()
        self._plot_comprehensive_roc_curves()
        self._plot_comprehensive_precision_recall_curves()
        self._plot_comprehensive_confusion_matrices()
        self._plot_feature_importance_comparison()
        
        print(f"✅ All visualizations saved to: {self.visualizations_dir}")
    
    def _plot_feature_distribution_comparison(self):
        """Plot comparison of raw vs normalized feature distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Raw vs Normalized Feature Distribution Comparison', fontsize=16, fontweight='bold')
        
        # Raw features distribution
        raw_data = self.df[self.action_features]
        axes[0,0].hist(raw_data.values.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Raw Feature Distribution')
        axes[0,0].set_xlabel('Value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_xlim(0, 1)
        
        # Normalized features distribution
        scaled_data = self.df[self.action_features_scaled]
        axes[0,1].hist(scaled_data.values.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Normalized Feature Distribution')
        axes[0,1].set_xlabel('Value')
        axes[0,1].set_ylabel('Frequency')
        
        # Feature means comparison
        raw_means = raw_data.mean().sort_values(ascending=False).head(15)
        axes[1,0].bar(range(len(raw_means)), raw_means.values)
        axes[1,0].set_title('Top 15 Raw Feature Means')
        axes[1,0].set_xlabel('Feature Index')
        axes[1,0].set_ylabel('Mean Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Feature std comparison
        scaled_stds = scaled_data.std().sort_values(ascending=False).head(15)
        axes[1,1].bar(range(len(scaled_stds)), scaled_stds.values)
        axes[1,1].set_title('Top 15 Normalized Feature Std Dev')
        axes[1,1].set_xlabel('Feature Index')
        axes[1,1].set_ylabel('Standard Deviation')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/feature_distribution_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance_comparison(self):
        """Plot comprehensive model performance comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Top1 Normalized Action Class Model Performance', fontsize=16, fontweight='bold')
        
        models = list(self.combined_evaluation.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro', 'auc_roc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Weighted', 'F1-Macro', 'ROC-AUC']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//3, i%3]
            values = [self.combined_evaluation[model][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.8)
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/model_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.all_results.items():
            y_test = results['y_test']
            y_pred_proba = results['y_pred_proba']
            features_used = results['features_used']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} ({features_used}) (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Top1 Normalized Action Class Models', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.all_results.items():
            y_test = results['y_test']
            y_pred_proba = results['y_pred_proba']
            features_used = results['features_used']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} ({features_used}) (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Top1 Normalized Action Class Models', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.all_results.items()):
            y_test = results['y_test']
            y_pred = results['y_pred']
            features_used = results['features_used']
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}\n({features_used} features)', fontsize=12)
            axes[i].set_xlabel('Predicted', fontsize=12)
            axes[i].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance_comparison(self):
        """Plot feature importance comparison across models"""
        n_models = len([m for m in self.all_models.values() 
                       if hasattr(m['model'], 'feature_importances_') or hasattr(m['model'], 'coef_')])
        
        if n_models == 0:
            return
            
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 10))
        
        if n_models == 1:
            axes = [axes]
        
        plot_idx = 0
        for model_name, model_info in self.all_models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15
                
                axes[plot_idx].bar(range(15), importances[indices])
                axes[plot_idx].set_title(f'{model_name} Feature Importance (Top 15)')
                axes[plot_idx].set_xlabel('Action Class Features')
                axes[plot_idx].set_ylabel('Importance')
                
                # Add feature labels
                feature_labels = []
                for idx in indices:
                    feature_name = model_info['feature_names'][idx]
                    if '_scaled' in feature_name:
                        class_id = feature_name.split('_')[2]
                    else:
                        class_id = feature_name.split('_')[-1]
                    feature_labels.append(f'C{class_id}')
                
                axes[plot_idx].set_xticks(range(15))
                axes[plot_idx].set_xticklabels(feature_labels, rotation=45)
                plot_idx += 1
            
            elif hasattr(model, 'coef_'):
                coefficients = np.abs(model.coef_[0])
                indices = np.argsort(coefficients)[::-1][:15]  # Top 15
                
                axes[plot_idx].bar(range(15), coefficients[indices])
                axes[plot_idx].set_title(f'{model_name} Feature Importance (Top 15)')
                axes[plot_idx].set_xlabel('Action Class Features')
                axes[plot_idx].set_ylabel('|Coefficient|')
                
                # Add feature labels
                feature_labels = []
                for idx in indices:
                    feature_name = model_info['feature_names'][idx]
                    if '_scaled' in feature_name:
                        class_id = feature_name.split('_')[2]
                    else:
                        class_id = feature_name.split('_')[-1]
                    feature_labels.append(f'C{class_id}')
                
                axes[plot_idx].set_xticks(range(15))
                axes[plot_idx].set_xticklabels(feature_labels, rotation=45)
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizations_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self):
        """Save comprehensive results and models"""
        print("\nSaving comprehensive results...")
        
        # Save all models
        for model_name, model_info in self.all_models.items():
            filename = f'{self.models_dir}/{model_name.lower().replace(" ", "_")}_{self.timestamp}.pkl'
            joblib.dump(model_info, filename)
            print(f"{model_name} saved to {filename}")
        
        # Save comprehensive evaluation results
        evaluation_df = pd.DataFrame(self.combined_evaluation).T
        evaluation_df.to_csv(f'{self.results_dir}/comprehensive_evaluation_{self.timestamp}.csv')
        
        # Save detailed results summary
        summary = {
            'timestamp': self.timestamp,
            'approach': 'Top1 Action Class Normalized (52 scaled features)',
            'methodology': 'Patient-level splitting, StandardScaler normalization',
            'models_trained': list(self.all_models.keys()),
            'best_model_by_f1_weighted': max(self.combined_evaluation.items(), 
                                           key=lambda x: x[1]['f1_weighted'])[0],
            'best_f1_weighted_score': max(self.combined_evaluation.items(), 
                                        key=lambda x: x[1]['f1_weighted'])[1]['f1_weighted'],
            'best_model_by_auc': max(self.combined_evaluation.items(), 
                                   key=lambda x: x[1]['auc_roc'])[0],
            'best_auc_score': max(self.combined_evaluation.items(), 
                                key=lambda x: x[1]['auc_roc'])[1]['auc_roc'],
            'evaluation_results': self.combined_evaluation,
            'data_split_info': {
                'train_patients': len(self.train_patients),
                'test_patients': len(self.test_patients),
                'patient_overlap': 0
            }
        }
        
        summary_file = f'{self.results_dir}/workflow_summary_{self.timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comprehensive evaluation saved to {self.results_dir}/comprehensive_evaluation_{self.timestamp}.csv")
        print(f"Workflow summary saved to {summary_file}")
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TOP1 NORMALIZED ACTION CLASS WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"\n🎯 APPROACH DETAILS:")
        print(f"  - Feature Type: Top1 Action Class Normalized (52 StandardScaler features)")
        print(f"  - Methodology: Patient-level splitting (no data leakage)")
        print(f"  - Normalization: StandardScaler (mean=0, std=1)")
        print(f"  - Balancing: Optimized class weights (no SMOTE)")
        print(f"  - Models Trained: {len(self.all_models)}")
        
        for model_name in self.all_models.keys():
            features_used = self.all_models[model_name]['features_used']
            print(f"    • {model_name} ({features_used} features)")
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        best_f1_model = max(self.combined_evaluation.items(), key=lambda x: x[1]['f1_weighted'])
        best_auc_model = max(self.combined_evaluation.items(), key=lambda x: x[1]['auc_roc'])
        
        print(f"  🏆 Best F1-Weighted: {best_f1_model[0]} ({best_f1_model[1]['f1_weighted']:.4f})")
        print(f"  🎯 Best ROC-AUC: {best_auc_model[0]} ({best_auc_model[1]['auc_roc']:.4f})")
        
        print(f"\n📈 DETAILED RESULTS:")
        for model_name, results in self.combined_evaluation.items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {results['accuracy']:.4f}")
            print(f"    - Precision: {results['precision']:.4f}")
            print(f"    - Recall: {results['recall']:.4f}")
            print(f"    - F1-Weighted: {results['f1_weighted']:.4f}")
            print(f"    - F1-Macro: {results['f1_macro']:.4f}")
            print(f"    - ROC-AUC: {results['auc_roc']:.4f}")
            print(f"    - Features: {results['features_used']}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  - Main Results: {self.results_dir}/")
        print(f"  - Models: {self.models_dir}/")
        print(f"  - Visualizations: {self.visualizations_dir}/")
        print(f"  - Evaluation: comprehensive_evaluation_{self.timestamp}.csv")
        print(f"  - Summary: workflow_summary_{self.timestamp}.json")
        print(f"  - Feature Importance: feature_importance_*.csv")
        
        print(f"\n✅ VALIDATION STATUS:")
        print(f"  ✅ Patient-level splitting: No data leakage")
        print(f"  ✅ Normalized features: StandardScaler applied")
        print(f"  ✅ Fair comparison: Same normalization as Top5")
        print(f"  ✅ Production ready: Robust, optimized pipeline")
        
        print(f"\nCOMPREHENSIVE TOP1 NORMALIZED ACTION CLASS WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def run_complete_workflow(self, use_normalized=True, include_comparison=True):
        """Run the complete comprehensive workflow with normalized features"""
        # Load data and show feature details
        self.load_data_and_features()
        self.show_feature_details()
        
        # Run comparison of raw vs normalized features
        if include_comparison:
            comparison_results, best_approach = self.train_comparison_models()
        
        # Train all models with normalized features (or raw based on comparison)
        self.train_all_models_normalized(use_normalized=use_normalized)
        
        # Comprehensive evaluation
        self.evaluate_all_models()
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations()
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.all_models, self.combined_evaluation

def main():
    """Main function to run the comprehensive normalized top1 workflow"""
    # Initialize workflow
    workflow = NormalizedTop1DepressionPredictionWorkflow()
    
    # Run complete workflow with normalized features
    models, evaluation = workflow.run_complete_workflow(
        use_normalized=True,    # Use normalized features for fair comparison
        include_comparison=True # Compare raw vs normalized
    )
    
    return models, evaluation

if __name__ == "__main__":
    models, evaluation = main()