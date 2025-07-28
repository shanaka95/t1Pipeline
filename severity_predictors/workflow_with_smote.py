"""
Comprehensive Depression Severity Prediction Workflow with SMOTE
This module provides a complete pipeline that compares models with and without SMOTE.
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# Import model classes
from base_severity_model import BaseSeverityModel
from xgb_severity_model import XGBoostSeverityModel
from random_forest_severity_model import RandomForestSeverityModel
from svm_severity_model import SVMSeverityModel

class SeverityWorkflowWithSMOTE:
    def __init__(self, processed_data_path='../processed_data/depression_processed.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the comprehensive severity workflow with SMOTE comparison"""
        self.processed_data_path = processed_data_path
        self.feature_info_path = feature_info_path
        self.df = None
        self.feature_info = None
        self.all_models = {}
        self.all_results = {}
        self.comparison_results = {}
        
    def load_data_and_features(self):
        """Load data and feature information"""
        print("Starting Comprehensive Depression Severity Prediction Workflow with SMOTE")
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
        
        # Show severity distribution
        severity_counts = self.df['Depression_3Class'].value_counts().sort_index()
        severity_labels = {1: 'Mild/Subclinical', 2: 'Moderate', 3: 'Severe'}
        print(f"\nSeverity Distribution:")
        for cls in severity_counts.index:
            label = severity_labels.get(cls, f'Class {cls}')
            percentage = (severity_counts[cls] / len(self.df)) * 100
            print(f"  {label} (Class {cls}): {severity_counts[cls]} ({percentage:.1f}%)")
        
        # Show class imbalance
        class_balance = severity_counts.max() / severity_counts.min()
        print(f"\nClass imbalance ratio: {class_balance:.2f}")
        if class_balance > 3:
            print("SIGNIFICANT CLASS IMBALANCE DETECTED - SMOTE will be beneficial")
        
        return self.df, self.feature_info
    
    def train_models_comparison(self, balance_methods=['none', 'smote']):
        """Train all models with different balance methods for comparison"""
        print("\n" + "="*60)
        print("TRAINING MODELS WITH SMOTE COMPARISON")
        print("="*60)
        
        # Initialize model trainers
        model_classes = {
            'XGBoost': XGBoostSeverityModel,
            'Random Forest': RandomForestSeverityModel,
            'SVM': SVMSeverityModel
        }
        
        # Train each model with different balance methods
        for balance_method in balance_methods:
            print(f"\n{'='*50}")
            print(f"TRAINING WITH BALANCE METHOD: {balance_method.upper()}")
            print(f"{'='*50}")
            
            for model_name, model_class in model_classes.items():
                print(f"\n{'-'*30} {model_name} {'-'*30}")
                
                # Initialize trainer
                trainer = model_class(self.processed_data_path, self.feature_info_path)
                
                # Load data and prepare features
                trainer.load_processed_data()
                X, y_3class, y_binary, feature_cols = trainer.prepare_features_targets()
                
                # Verify feature integrity
                trainer.verify_feature_integrity(feature_cols)
                
                # Split data
                X_train, X_test, y_train, y_test = trainer.split_data(X, y_3class)
                
                # Train model with specific balance method
                model_key = f"{model_name}_{balance_method}"
                
                if model_name == 'XGBoost':
                    model = trainer.train_xgboost_model(
                        X_train, y_train, X_test, y_test, 
                        model_name=f"xgb_{balance_method}",
                        tune_hyperparameters=False,
                        balance_method=balance_method,
                        use_class_weights=False
                    )
                elif model_name == 'Random Forest':
                    model = trainer.train_random_forest_model(
                        X_train, y_train, X_test, y_test, 
                        model_name=f"rf_{balance_method}",
                        tune_hyperparameters=False,
                        balance_method=balance_method,
                        use_class_weights=False
                    )
                elif model_name == 'SVM':
                    model = trainer.train_svm_model(
                        X_train, y_train, X_test, y_test, 
                        model_name=f"svm_{balance_method}",
                        tune_hyperparameters=False,
                        balance_method=balance_method,
                        use_class_weights=False
                    )
                
                # Store results
                self.all_models[model_key] = trainer.models
                self.all_results[model_key] = trainer.results
                
                print(f"{model_name} with {balance_method} completed!")
        
        print(f"\nAll models trained successfully!")
        print(f"Total model configurations: {len(self.all_models)}")
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all models with SMOTE comparison"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION WITH SMOTE COMPARISON")
        print("="*60)
        
        evaluation_results = {}
        
        for config_name, model_results in self.all_results.items():
            for sub_model_name, results in model_results.items():
                full_name = f"{config_name}_{sub_model_name}"
                
                y_test = results['y_test']
                y_pred = results['y_pred']
                y_pred_proba = results['y_pred_proba']
                balance_method = results.get('balance_method', 'none')
                
                print(f"\n{full_name.upper()} Results:")
                print(f"Balance method: {balance_method}")
                print("-" * 50)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Multi-class AUC calculation
                try:
                    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                    if y_test_bin.shape[1] == 3:
                        auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.0
                except:
                    auc = 0.0
                
                # Store results
                evaluation_results[full_name] = {
                    'model_type': config_name.split('_')[0],
                    'balance_method': balance_method,
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
        
        self.comparison_results = evaluation_results
        return evaluation_results
    
    def create_smote_comparison_visualizations(self):
        """Create visualizations comparing SMOTE vs no-SMOTE performance"""
        print("\nCreating SMOTE comparison visualizations...")
        
        # Create output directory
        os.makedirs('../severity_results', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. SMOTE vs No-SMOTE Comparison
        self._plot_smote_comparison()
        
        # 2. Model Performance by Balance Method
        self._plot_balance_method_comparison()
        
        # 3. Confusion Matrices Comparison
        self._plot_confusion_matrices_comparison()
        
        print("SMOTE comparison visualizations saved to '../severity_results/' directory")
    
    def _plot_smote_comparison(self):
        """Plot SMOTE vs no-SMOTE performance comparison"""
        # Prepare data for comparison
        comparison_data = []
        
        for model_config, results in self.comparison_results.items():
            model_type = results['model_type']
            balance_method = results['balance_method']
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                comparison_data.append({
                    'Model': model_type,
                    'Balance Method': 'Without SMOTE' if balance_method == 'none' else 'With SMOTE',
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': results[metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, metric in enumerate(metrics):
            metric_data = comparison_df[comparison_df['Metric'] == metric]
            sns.barplot(data=metric_data, x='Model', y='Score', hue='Balance Method', ax=axes[i])
            axes[i].set_title(f'{metric} Comparison: SMOTE vs No-SMOTE')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            if i == 0:
                axes[i].legend(title='Balance Method')
            else:
                axes[i].get_legend().remove()
        
        plt.tight_layout()
        plt.savefig('../severity_results/smote_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_balance_method_comparison(self):
        """Plot detailed balance method comparison"""
        # Create summary table
        summary_data = []
        
        for model_config, results in self.comparison_results.items():
            summary_data.append({
                'Model': results['model_type'],
                'Balance Method': results['balance_method'],
                'F1-Score': results['f1_score'],
                'Accuracy': results['accuracy']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Pivot for easier comparison
        pivot_f1 = summary_df.pivot(index='Model', columns='Balance Method', values='F1-Score')
        pivot_acc = summary_df.pivot(index='Model', columns='Balance Method', values='Accuracy')
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # F1-Score comparison
        pivot_f1.plot(kind='bar', ax=axes[0], color=['lightcoral', 'lightblue'])
        axes[0].set_title('F1-Score Comparison by Balance Method')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_xlabel('Model Type')
        axes[0].legend(title='Balance Method')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        pivot_acc.plot(kind='bar', ax=axes[1], color=['lightcoral', 'lightblue'])
        axes[1].set_title('Accuracy Comparison by Balance Method')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Model Type')
        axes[1].legend(title='Balance Method')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../severity_results/balance_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices_comparison(self):
        """Plot confusion matrices for SMOTE vs no-SMOTE"""
        # Get best performing model with and without SMOTE
        smote_results = {k: v for k, v in self.comparison_results.items() if 'smote' in k}
        none_results = {k: v for k, v in self.comparison_results.items() if 'none' in k}
        
        if smote_results and none_results:
            best_smote = max(smote_results.items(), key=lambda x: x[1]['f1_score'])
            best_none = max(none_results.items(), key=lambda x: x[1]['f1_score'])
            
            # Find corresponding confusion matrices
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, (best_config, title_suffix) in enumerate([(best_none, 'Without SMOTE'), (best_smote, 'With SMOTE')]):
                config_name = best_config[0]
                
                # Find the corresponding results
                for model_results in self.all_results.values():
                    for sub_model_name, results in model_results.items():
                        if config_name in f"{model_results}_{sub_model_name}":
                            y_test = results['y_test']
                            y_pred = results['y_pred']
                            
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                                       xticklabels=['Mild', 'Moderate', 'Severe'],
                                       yticklabels=['Mild', 'Moderate', 'Severe'])
                            
                            model_name = best_config[1]['model_type']
                            f1_score = best_config[1]['f1_score']
                            axes[i].set_title(f'Best {model_name} {title_suffix}\n(F1: {f1_score:.3f})')
                            axes[i].set_xlabel('Predicted Severity')
                            axes[i].set_ylabel('Actual Severity')
                            break
            
            plt.tight_layout()
            plt.savefig('../severity_results/confusion_matrices_smote_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_smote_impact_report(self):
        """Generate detailed report on SMOTE impact"""
        print("\n" + "="*60)
        print("SMOTE IMPACT ANALYSIS REPORT")
        print("="*60)
        
        # Calculate improvements
        improvements = {}
        
        for model_type in ['XGBoost', 'Random Forest', 'SVM']:
            none_key = None
            smote_key = None
            
            for config_name, results in self.comparison_results.items():
                if results['model_type'] == model_type:
                    if results['balance_method'] == 'none':
                        none_key = config_name
                    elif results['balance_method'] == 'smote':
                        smote_key = config_name
            
            if none_key and smote_key:
                none_results = self.comparison_results[none_key]
                smote_results = self.comparison_results[smote_key]
                
                improvements[model_type] = {
                    'accuracy_improvement': smote_results['accuracy'] - none_results['accuracy'],
                    'f1_improvement': smote_results['f1_score'] - none_results['f1_score'],
                    'precision_improvement': smote_results['precision'] - none_results['precision'],
                    'recall_improvement': smote_results['recall'] - none_results['recall'],
                    'none_f1': none_results['f1_score'],
                    'smote_f1': smote_results['f1_score'],
                    'none_accuracy': none_results['accuracy'],
                    'smote_accuracy': smote_results['accuracy']
                }
        
        # Print report
        print(f"\nSMOTE Impact Summary:")
        print(f"{'Model':<15} {'F1 Without':<12} {'F1 With':<12} {'F1 Improvement':<15} {'Accuracy Improvement':<20}")
        print("-" * 80)
        
        total_f1_improvement = 0
        total_acc_improvement = 0
        
        for model_type, improvements_data in improvements.items():
            f1_imp = improvements_data['f1_improvement']
            acc_imp = improvements_data['accuracy_improvement']
            
            print(f"{model_type:<15} {improvements_data['none_f1']:<12.4f} {improvements_data['smote_f1']:<12.4f} "
                  f"{f1_imp:<15.4f} {acc_imp:<20.4f}")
            
            total_f1_improvement += f1_imp
            total_acc_improvement += acc_imp
        
        avg_f1_improvement = total_f1_improvement / len(improvements)
        avg_acc_improvement = total_acc_improvement / len(improvements)
        
        print("-" * 80)
        print(f"{'AVERAGE':<15} {'':<12} {'':<12} {avg_f1_improvement:<15.4f} {avg_acc_improvement:<20.4f}")
        
        # Best performing configurations
        print(f"\nBest Performing Configurations:")
        best_overall = max(self.comparison_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"Overall Best: {best_overall[1]['model_type']} with {best_overall[1]['balance_method']} (F1: {best_overall[1]['f1_score']:.4f})")
        
        smote_models = {k: v for k, v in self.comparison_results.items() if v['balance_method'] == 'smote'}
        if smote_models:
            best_smote = max(smote_models.items(), key=lambda x: x[1]['f1_score'])
            print(f"Best with SMOTE: {best_smote[1]['model_type']} (F1: {best_smote[1]['f1_score']:.4f})")
        
        none_models = {k: v for k, v in self.comparison_results.items() if v['balance_method'] == 'none'}
        if none_models:
            best_none = max(none_models.items(), key=lambda x: x[1]['f1_score'])
            print(f"Best without SMOTE: {best_none[1]['model_type']} (F1: {best_none[1]['f1_score']:.4f})")
        
        return improvements
    
    def save_comprehensive_results(self):
        """Save comprehensive results and comparison"""
        print("\nSaving comprehensive results...")
        
        os.makedirs('../saved_models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all models
        for config_name, models in self.all_models.items():
            for model_key, model in models.items():
                filename = f'../saved_models/severity_{config_name}_{model_key}_{timestamp}.pkl'
                joblib.dump(model, filename)
                print(f"Severity {config_name} {model_key} saved to {filename}")
        
        # Save comparison results
        comparison_df = pd.DataFrame(self.comparison_results).T
        comparison_df.to_csv(f'../saved_models/severity_smote_comparison_{timestamp}.csv')
        print(f"SMOTE comparison results saved to ../saved_models/severity_smote_comparison_{timestamp}.csv")
        
        # Save detailed summary
        summary = {
            'timestamp': timestamp,
            'models_compared': list(self.all_models.keys()),
            'balance_methods': ['none', 'smote'],
            'best_overall': max(self.comparison_results.items(), key=lambda x: x[1]['f1_score'])[0],
            'comparison_results': self.comparison_results
        }
        
        summary_file = f'../saved_models/severity_smote_workflow_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Workflow summary saved to {summary_file}")
    
    def run_complete_workflow(self):
        """Run the complete comprehensive workflow with SMOTE comparison"""
        # Load data and show details
        self.load_data_and_features()
        
        # Train all models with comparison
        self.train_models_comparison(balance_methods=['none', 'smote'])
        
        # Evaluate all models
        self.evaluate_all_models()
        
        # Create SMOTE comparison visualizations
        self.create_smote_comparison_visualizations()
        
        # Generate impact report
        improvements = self.generate_smote_impact_report()
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        # Final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE SEVERITY WORKFLOW WITH SMOTE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\nModels Trained: {len(self.all_models)} configurations")
        print(f"Balance Methods: none, smote")
        
        best_model = max(self.comparison_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"Best Overall Model: {best_model[1]['model_type']} with {best_model[1]['balance_method']} (F1: {best_model[1]['f1_score']:.4f})")
        
        print(f"\nOutput Files:")
        print(f"  - Models: ../saved_models/ directory")
        print(f"  - Visualizations: ../severity_results/ directory")
        print(f"  - SMOTE comparison: severity_smote_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        return self.all_models, self.comparison_results, improvements

def main():
    """Main function to run comprehensive severity prediction workflow with SMOTE"""
    # Initialize workflow
    workflow = SeverityWorkflowWithSMOTE()
    
    # Run complete workflow
    models, comparison_results, improvements = workflow.run_complete_workflow()
    
    return models, comparison_results, improvements

if __name__ == "__main__":
    models, comparison_results, improvements = main() 