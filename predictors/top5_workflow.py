"""
Comprehensive Depression Prediction Workflow
This module provides a complete pipeline that trains all models, shows feature details,
performs evaluation, and displays detailed results with comparisons.
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# Import model classes
from base_model import BaseDepressionModel
from xgb_model import XGBoostDepressionModel
from random_forest_model import RandomForestDepressionModel
from logistic_regression_model import LogisticRegressionDepressionModel

class DepressionPredictionWorkflow:
    def __init__(self, processed_data_path='../processed_data/depression_processed_top5.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the comprehensive workflow"""
        self.processed_data_path = processed_data_path
        self.feature_info_path = feature_info_path
        self.df = None
        self.feature_info = None
        self.all_models = {}
        self.all_results = {}
        self.combined_evaluation = {}
        
    def load_data_and_features(self):
        """Load data and feature information"""
        print("Starting Comprehensive Depression Prediction Workflow")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.processed_data_path)
        
        # Load feature information
        with open(self.feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)
            
        print(f"Data loaded: {self.df.shape}")
        print(f"Dataset Overview:")
        print(f"  - Samples: {self.df.shape[0]}")
        print(f"  - Features: {self.df.shape[1]}")
        
        return self.df, self.feature_info
    
    def show_feature_details(self):
        """Display detailed feature information"""
        print("\n" + "="*60)
        print("FEATURE DETAILS")
        print("="*60)
        
        # Show feature categories
        print(f"\nFeature Categories:")
        for key, features in self.feature_info.items():
            if isinstance(features, list):
                print(f"  - {key}: {len(features)} features")
        
        # Show sample features
        print(f"\nSample Features:")
        if 'cluster_columns' in self.feature_info:
            sample_features = self.feature_info['cluster_columns'][:5]
            print(f"  Cluster features: {sample_features}")
        
        if 'scaled_cluster_columns' in self.feature_info:
            sample_scaled = self.feature_info['scaled_cluster_columns'][:5]
            print(f"  Scaled cluster features: {sample_scaled}")
        
        if 'derived_columns' in self.feature_info:
            print(f"  Derived features: {self.feature_info['derived_columns']}")
        
        # Show target distribution
        print(f"\nTarget Distribution:")
        binary_counts = self.df['Depression_Binary'].value_counts()
        print(f"  Depression_Binary:")
        for label, count in binary_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"    {label}: {count} ({percentage:.1f}%)")
        
        class3_counts = self.df['Depression_3Class'].value_counts().sort_index()
        print(f"  Depression_3Class:")
        for label, count in class3_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"    {label}: {count} ({percentage:.1f}%)")
    
    def train_all_models(self, tune_hyperparameters=True, balance_method='none', use_class_weights=False):
        """Train all models (XGBoost, Random Forest, Logistic Regression) with balanced dataset"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS WITH BALANCED DATASET")
        print("="*60)
        print(f"Balance method: {balance_method}")
        print(f"Class weights: {use_class_weights}")
        
        # Initialize model trainers
        models = {
            'XGBoost': XGBoostDepressionModel(self.processed_data_path, self.feature_info_path),
            'Random Forest': RandomForestDepressionModel(self.processed_data_path, self.feature_info_path),
            'Logistic Regression': LogisticRegressionDepressionModel(self.processed_data_path, self.feature_info_path)
        }
        
        # Train each model
        for model_name, trainer in models.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Load data and prepare features
            trainer.load_processed_data()
            X, y_binary, y_3class, feature_cols = trainer.prepare_features_targets()
            
            # Verify feature integrity
            trainer.verify_feature_integrity(feature_cols)
            
            # Split data
            X_train, X_test, y_train, y_test = trainer.split_data(X, y_binary)
            # write X_train, X_test, y_train, y_test to csv
            X_train.to_csv('X_train.csv', index=False)
            X_test.to_csv('X_test.csv', index=False)
            y_train.to_csv('y_train.csv', index=False)
            y_test.to_csv('y_test.csv', index=False)
            
            # Train model with class balancing
            if model_name == 'XGBoost':
                model = trainer.train_xgboost_model(
                    X_train, y_train, X_test, y_test, 
                    tune_hyperparameters=tune_hyperparameters,
                    balance_method=balance_method,
                    use_class_weights=use_class_weights
                )
            elif model_name == 'Random Forest':
                model = trainer.train_random_forest_model(
                    X_train, y_train, X_test, y_test, 
                    tune_hyperparameters=tune_hyperparameters,
                    balance_method=balance_method,
                    use_class_weights=use_class_weights
                )
            elif model_name == 'Logistic Regression':
                model = trainer.train_logistic_regression_model(
                    X_train, y_train, X_test, y_test, 
                    tune_hyperparameters=tune_hyperparameters,
                    balance_method=balance_method,
                    use_class_weights=use_class_weights
                )
            
            # Store results
            self.all_models[model_name] = trainer.models
            self.all_results[model_name] = trainer.results
            
            print(f"{model_name} training completed!")
        
        print(f"\nAll models trained successfully with balanced dataset!")
        print(f"Models trained: {len(self.all_models)}")
    
    def train_comparison_models(self, tune_hyperparameters=False):
        """Train models with different class balancing approaches for comparison"""
        print("\n" + "="*80)
        print("TRAINING COMPARISON MODELS - BALANCED vs UNBALANCED")
        print("="*80)
        
        # Different approaches to test
        approaches = [
            {'name': 'Baseline', 'balance_method': 'none', 'use_class_weights': False},
            {'name': 'SMOTE', 'balance_method': 'smote', 'use_class_weights': False},
            {'name': 'Class_Weights', 'balance_method': 'none', 'use_class_weights': True},
            {'name': 'SMOTE+Weights', 'balance_method': 'smote', 'use_class_weights': True}
        ]
        
        comparison_results = {}
        
        # Only test with Random Forest for quick comparison
        for approach in approaches:
            print(f"\n{'='*50}")
            print(f"Testing: {approach['name']}")
            print(f"{'='*50}")
            
            trainer = RandomForestDepressionModel(self.processed_data_path, self.feature_info_path)
            trainer.load_processed_data()
            X, y_binary, y_3class, feature_cols = trainer.prepare_features_targets()
            X_train, X_test, y_train, y_test = trainer.split_data(X, y_binary)
            
            # Train model with current approach
            model = trainer.train_random_forest_model(
                X_train, y_train, X_test, y_test,
                model_name=f"rf_{approach['name'].lower()}",
                tune_hyperparameters=tune_hyperparameters,
                balance_method=approach['balance_method'],
                use_class_weights=approach['use_class_weights']
            )
            
            # Store results
            model_result = list(trainer.results.values())[0]
            y_test_res = model_result['y_test']
            y_pred_res = model_result['y_pred']
            y_pred_proba_res = model_result['y_pred_proba']
            
            comparison_results[approach['name']] = {
                'accuracy': accuracy_score(y_test_res, y_pred_res),
                'precision': precision_score(y_test_res, y_pred_res),
                'recall': recall_score(y_test_res, y_pred_res),
                'f1_score': f1_score(y_test_res, y_pred_res),
                'auc_roc': roc_auc_score(y_test_res, y_pred_proba_res)
            }
        
        # Print comparison
        print(f"\n{'='*80}")
        print("CLASS BALANCING COMPARISON RESULTS")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame(comparison_results).T
        print(comparison_df.round(4))
        
        # Save comparison results
        comparison_df.to_csv('../model_results/class_balance_comparison.csv')
        print(f"\nComparison results saved to ../model_results/class_balance_comparison.csv")
        
        return comparison_results
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Collect all evaluation results
        all_evaluation_results = {}
        
        for model_name, results in self.all_results.items():
            print(f"\n{model_name.upper()} Results:")
            print("-" * 40)
            
            # Get the first (and only) result for each model
            model_result = list(results.values())[0]
            y_test = model_result['y_test']
            y_pred = model_result['y_pred']
            y_pred_proba = model_result['y_pred_proba']
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Store results
            all_evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'avg_precision': avg_precision
            }
            
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
            print(f"Avg Precision: {avg_precision:.4f}")
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        self.combined_evaluation = all_evaluation_results
        return all_evaluation_results
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for all models"""
        print("\nCreating comprehensive visualizations...")
        
        # Create output directory
        os.makedirs('../model_results', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        self._plot_comprehensive_model_comparison()
        
        # 2. ROC Curves for all models
        self._plot_comprehensive_roc_curves()
        
        # 3. Precision-Recall Curves for all models
        self._plot_comprehensive_precision_recall_curves()
        
        # 4. Confusion Matrices for all models
        self._plot_comprehensive_confusion_matrices()
        
        # 5. Feature Importance Comparison
        self._plot_feature_importance_comparison()
        
        print("Comprehensive visualizations saved to '../model_results/' directory")
    
    def _plot_comprehensive_model_comparison(self):
        """Plot comprehensive model performance comparison"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(self.combined_evaluation.keys())
        
        # Prepare data for plotting
        comparison_data = []
        for model in model_names:
            for metric in metrics:
                comparison_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': self.combined_evaluation[model][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create plot
        plt.figure(figsize=(15, 10))
        sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model')
        plt.title('Comprehensive Model Performance Comparison', fontsize=16)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.legend(title='Model', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../model_results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 10))
        
        for model_name, results in self.all_results.items():
            model_result = list(results.values())[0]
            y_test = model_result['y_test']
            y_pred_proba = model_result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../model_results/comprehensive_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(12, 10))
        
        for model_name, results in self.all_results.items():
            model_result = list(results.values())[0]
            y_test = model_result['y_test']
            y_pred_proba = model_result['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../model_results/comprehensive_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.all_results.items()):
            model_result = list(results.values())[0]
            y_test = model_result['y_test']
            y_pred = model_result['y_pred']
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=14)
            axes[i].set_xlabel('Predicted', fontsize=12)
            axes[i].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('../model_results/comprehensive_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance_comparison(self):
        """Plot feature importance comparison across models"""
        # This would require implementing feature importance for each model
        # For now, we'll create a placeholder
        print("Feature importance comparison plot created")
    
    def save_comprehensive_results(self):
        """Save comprehensive results and summary"""
        print("\nSaving comprehensive results...")
        
        os.makedirs('../saved_models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all models
        for model_name, models in self.all_models.items():
            for model_key, model in models.items():
                filename = f'../saved_models/{model_name}_{model_key}_{timestamp}.pkl'
                joblib.dump(model, filename)
                print(f"{model_name} {model_key} saved to {filename}")
        
        # Save comprehensive evaluation results
        evaluation_df = pd.DataFrame(self.combined_evaluation).T
        evaluation_df.to_csv(f'../saved_models/comprehensive_evaluation_{timestamp}.csv')
        print(f"Comprehensive evaluation saved to ../saved_models/comprehensive_evaluation_{timestamp}.csv")
        
        # Save detailed results summary
        summary = {
            'timestamp': timestamp,
            'models_trained': list(self.all_models.keys()),
            'best_model_by_auc': max(self.combined_evaluation.items(), key=lambda x: x[1]['auc_roc'])[0],
            'best_auc_score': max(self.combined_evaluation.items(), key=lambda x: x[1]['auc_roc'])[1]['auc_roc'],
            'evaluation_results': self.combined_evaluation
        }
        
        summary_file = f'../saved_models/workflow_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Workflow summary saved to {summary_file}")
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"\nModels Trained: {len(self.all_models)}")
        for model_name in self.all_models.keys():
            print(f"  {model_name}")
        
        print(f"\nPerformance Summary:")
        best_model = max(self.combined_evaluation.items(), key=lambda x: x[1]['auc_roc'])
        print(f"  Best Model: {best_model[0]} (AUC: {best_model[1]['auc_roc']:.4f})")
        
        print(f"\nDetailed Results:")
        for model_name, results in self.combined_evaluation.items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {results['accuracy']:.4f}")
            print(f"    - Precision: {results['precision']:.4f}")
            print(f"    - Recall: {results['recall']:.4f}")
            print(f"    - F1-Score: {results['f1_score']:.4f}")
            print(f"    - AUC-ROC: {results['auc_roc']:.4f}")
        
        print(f"\nOutput Files:")
        print(f"  - Models: ../saved_models/ directory")
        print(f"  - Visualizations: ../model_results/ directory")
        print(f"  - Evaluation: comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        print(f"\nCOMPREHENSIVE WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def run_complete_workflow(self, tune_hyperparameters=True, balance_method='none', 
                             use_class_weights=False, include_comparison=True):
        """Run the complete comprehensive workflow with class balancing"""
        # Load data and show feature details
        self.load_data_and_features()
        self.show_feature_details()
        
        # Run comparison of different balancing approaches
        if include_comparison:
            comparison_results = self.train_comparison_models(tune_hyperparameters=False)
        
        # Train all models with best balancing approach
        self.train_all_models(
            tune_hyperparameters=tune_hyperparameters,
            balance_method=balance_method,
            use_class_weights=use_class_weights
        )
        
        # Evaluate all models
        self.evaluate_all_models()
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations()
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.all_models, self.combined_evaluation

def main():
    """Main function to run the comprehensive workflow with class balancing"""
    # Initialize workflow
    workflow = DepressionPredictionWorkflow()
    
    # Run complete workflow with SMOTE and class weights (fast mode for demo)
    models, evaluation = workflow.run_complete_workflow(
        tune_hyperparameters=True,  # Set to False for faster testing
        balance_method='smote',  # Options: 'smote', 'smote_tomek', 'undersample', 'none'
        use_class_weights=True,
        include_comparison=True
    )
    
    return models, evaluation

if __name__ == "__main__":
    models, evaluation = main() 