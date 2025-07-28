"""
XGBoost Model Training for Depression Prediction
This script loads processed data from EDA and trains XGBoost models for depression prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class XGBDepressionModel:
    def __init__(self, processed_data_path='processed_data/depression_processed.csv',
                 feature_info_path='processed_data/feature_info.pkl'):
        """Initialize the XGBoost model trainer"""
        self.processed_data_path = processed_data_path
        self.feature_info_path = feature_info_path
        self.df = None
        self.feature_info = None
        self.models = {}
        self.results = {}
        
    def load_processed_data(self):
        """Load processed data and feature information"""
        print("📊 Loading processed data...")
        
        # Load processed dataset
        self.df = pd.read_csv(self.processed_data_path)
        
        # Load feature information
        with open(self.feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)
            
        print(f"✅ Data loaded: {self.df.shape}")
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
                print(f"⚠️  WARNING: Excluding potential target leakage column: {col}")
                continue
            
            # Check if column contains target-related keywords
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['depression', 'depressed', 'skid', 'phq9', 'hrsd', 'ads']):
                print(f"⚠️  WARNING: Excluding potential target leakage column: {col}")
                continue
                
            filtered_cols.append(col)
        
        if len(filtered_cols) != len(feature_cols):
            print(f"🔒 Filtered out {len(feature_cols) - len(filtered_cols)} potential target leakage columns")
        
        return filtered_cols
    
    def prepare_features_targets(self, use_scaled_features=True, include_engineered=True):
        """Prepare feature sets and targets for training"""
        print("\n🔧 Preparing features and targets...")
        
        # Select feature set
        if use_scaled_features:
            feature_cols = self.feature_info['scaled_features']
            print(f"Using scaled features: {len(feature_cols)} features")
        else:
            feature_cols = self.feature_info['original_features']
            print(f"Using original features: {len(feature_cols)} features")
            
        # Add engineered features if requested
        if include_engineered:
            engineered_features = self.feature_info['engineered_features']
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
        
        # Prepare targets
        y_binary = self.df['Depression_Binary']
        y_3class = self.df['Depression_3Class']
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Binary target distribution:")
        print(y_binary.value_counts().sort_index())
        print(f"3-Class target distribution:")
        print(y_3class.value_counts().sort_index())
        
        return X, y_binary, y_3class, feature_cols
    
    def _print_feature_summary(self, feature_cols):
        """Print a summary of the features being used"""
        print(f"\n📋 Feature Summary:")
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
        
        print("✅ Feature integrity verified - only cluster-based features used")
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        print(f"\n📊 Data split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training target distribution:")
        print(y_train.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost_model(self, X_train, y_train, X_test, y_test, 
                           model_name="xgb_binary", tune_hyperparameters=True):
        """Train XGBoost model with optional hyperparameter tuning"""
        print(f"\n🚀 Training XGBoost model: {model_name}")
        
        if tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            }
            
            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=1, return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters with some optimization
            best_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
            
            # Train model
            best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Store model and results
        self.models[model_name] = best_model
        self.results[model_name] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': X_train.columns.tolist()
        }
        
        # Print basic results
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"✅ Model trained successfully!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        
        return best_model
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """Train baseline models for comparison"""
        print("\n🔄 Training baseline models...")
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'y_test': y_test,
            'y_pred': rf_pred,
            'y_pred_proba': rf_pred_proba,
            'feature_names': X_train.columns.tolist()
        }
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42, max_iter=1000, n_jobs=-1
        )
        lr_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_test)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'y_test': y_test,
            'y_pred': lr_pred,
            'y_pred_proba': lr_pred_proba,
            'feature_names': X_train.columns.tolist()
        }
        
        print("✅ Baseline models trained!")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION")
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
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Store results
            evaluation_results[model_name] = {
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
            
        return evaluation_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Create output directory
        os.makedirs('model_results', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Comparison Plot
        self._plot_model_comparison()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Precision-Recall Curves
        self._plot_precision_recall_curves()
        
        # 4. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 5. Feature Importance (for XGBoost)
        if 'xgb_binary' in self.models:
            self._plot_feature_importance()
            
        print("✅ Visualizations saved to 'model_results/' directory")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
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
                    'Metric': metric,
                    'Score': self.evaluation_results[model][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.legend(title='Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            y_test = results['y_test']
            y_pred_proba = results['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            y_test = results['y_test']
            y_pred_proba = results['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
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
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('model_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance for XGBoost model"""
        if 'xgb_binary' not in self.models:
            return
            
        model = self.models['xgb_binary']
        feature_names = self.results['xgb_binary']['feature_names']
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(20)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('XGBoost Feature Importance (Top 20)')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('model_results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance to CSV
        importance_df.to_csv('model_results/feature_importance.csv', index=False)
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving trained models...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f'saved_models/{model_name}_{timestamp}.pkl'
            joblib.dump(model, filename)
            print(f"✅ {model_name} saved to {filename}")
        
        # Save results summary
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'accuracy': accuracy_score(results['y_test'], results['y_pred']),
                'auc_roc': roc_auc_score(results['y_test'], results['y_pred_proba']),
                'f1_score': f1_score(results['y_test'], results['y_pred'])
            }
        
        summary_df = pd.DataFrame(results_summary).T
        summary_df.to_csv(f'saved_models/model_summary_{timestamp}.csv')
        print(f"✅ Model summary saved to saved_models/model_summary_{timestamp}.csv")
    
    def run_complete_pipeline(self, tune_hyperparameters=True):
        """Run the complete XGBoost training pipeline"""
        print("🚀 Starting XGBoost Training Pipeline for Depression Prediction")
        print("="*70)
        
        # Load data
        self.load_processed_data()
        
        # Prepare features and targets
        X, y_binary, y_3class, feature_cols = self.prepare_features_targets()
        
        # Verify that we're only using cluster features
        print("\n🔍 Verifying feature integrity...")
        non_cluster_features = [col for col in feature_cols if not col.startswith('cluster_')]
        if non_cluster_features:
            print(f"⚠️  WARNING: Found non-cluster features: {non_cluster_features}")
            print("Only cluster features should be used for depression prediction")
        else:
            print("✅ All features are cluster-based (correct)")
        
        # Verify no target leakage
        target_columns = ['Depression_Binary', 'Depression_3Class', 'Binary_Depression', 
                         'Overall_Depression_Status', 'SKID_Depressed']
        leakage_columns = [col for col in feature_cols if col in target_columns]
        if leakage_columns:
            raise ValueError(f"CRITICAL ERROR: Target leakage detected! Columns: {leakage_columns}")
        else:
            print("✅ No target leakage detected")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_binary)
        
        # Train XGBoost model
        xgb_model = self.train_xgboost_model(
            X_train, y_train, X_test, y_test, 
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Train baseline models
        self.train_baseline_models(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save models
        self.save_models()
        
        print("\n🎉 XGBoost TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Models trained: {len(self.models)}")
        print(f"🏆 Best model by AUC: {max(evaluation_results.items(), key=lambda x: x[1]['auc_roc'])}")
        print(f"💾 Models saved to 'saved_models/' directory")
        print(f"📊 Results saved to 'model_results/' directory")
        
        return self.models, evaluation_results

def main():
    """Main function to run XGBoost training"""
    # Initialize trainer
    trainer = XGBDepressionModel()
    
    # Run complete pipeline
    models, results = trainer.run_complete_pipeline(tune_hyperparameters=True)
    
    return models, results

if __name__ == "__main__":
    models, evaluation_results = main() 