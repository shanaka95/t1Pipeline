"""
Random Forest Model for Depression Severity Prediction
This module implements Random Forest for 3-class severity classification with SMOTE support.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from base_severity_model import BaseSeverityModel

class RandomForestSeverityModel(BaseSeverityModel):
    def __init__(self, processed_data_path='../processed_data/depression_processed.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the Random Forest severity model trainer"""
        super().__init__(processed_data_path, feature_info_path)
        
    def train_random_forest_model(self, X_train, y_train, X_test, y_test, 
                                model_name="rf_severity", tune_hyperparameters=True, 
                                balance_method='smote', use_class_weights=False):
        """Train Random Forest model for severity prediction with SMOTE support"""
        print(f"\nTraining Random Forest severity model: {model_name}")
        print(f"Balance method: {balance_method}")
        print(f"Class weights: {use_class_weights}")
        
        # Handle class imbalance with SMOTE
        if balance_method != 'none':
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train, y_train, method=balance_method
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Set class weights
        class_weight = 'balanced' if use_class_weights else None
        if use_class_weights:
            print(f"Using class_weight: {class_weight}")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid (reduced for faster training on multi-class)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            # Create Random Forest classifier with class weights
            rf_model = RandomForestClassifier(
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                rf_model, param_grid, cv=cv, scoring='f1_weighted',
                n_jobs=-1, verbose=1, return_train_score=True
            )
            
            grid_search.fit(X_train_balanced, y_train_balanced)
            
            # Best model
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters with optimization
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            best_model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions on original test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Store model and results
        self.models[model_name] = best_model
        self.results[model_name] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': X_train.columns.tolist(),
            'balance_method': balance_method,
            'use_class_weights': use_class_weights
        }
        
        # Print basic results
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"Model trained successfully!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        return best_model
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest severity model"""
        if 'rf_severity' not in self.models:
            return
            
        model = self.models['rf_severity']
        feature_names = self.results['rf_severity']['feature_names']
        
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
        plt.title('Random Forest Feature Importance (Top 20) - Severity Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('../severity_results/rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance to CSV
        importance_df.to_csv('../severity_results/rf_feature_importance.csv', index=False)
        print("Random Forest feature importance plot and CSV saved")
    
    def run_training_pipeline(self, tune_hyperparameters=True, balance_method='smote', use_class_weights=False):
        """Run the complete Random Forest severity training pipeline with SMOTE"""
        print("Starting Random Forest Severity Training Pipeline with SMOTE")
        print("="*70)
        
        # Load data
        self.load_processed_data()
        
        # Prepare features and targets
        X, y_3class, y_binary, feature_cols = self.prepare_features_targets()
        
        # Verify feature integrity
        self.verify_feature_integrity(feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_3class)
        
        # Train Random Forest model with SMOTE
        rf_model = self.train_random_forest_model(
            X_train, y_train, X_test, y_test, 
            tune_hyperparameters=tune_hyperparameters,
            balance_method=balance_method,
            use_class_weights=use_class_weights
        )
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Save models
        self.save_models()
        
        print("\nRandom Forest Severity TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Models trained: {len(self.models)}")
        print(f"Best model by F1: {max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])}")
        print(f"Models saved to '../saved_models/' directory")
        print(f"Results saved to '../severity_results/' directory")
        
        return self.models, evaluation_results

def main():
    """Main function to run Random Forest severity training with SMOTE"""
    # Initialize trainer
    trainer = RandomForestSeverityModel()
    
    # Run complete pipeline with SMOTE
    models, results = trainer.run_training_pipeline(
        tune_hyperparameters=False,  # Set to True for full hyperparameter tuning
        balance_method='smote',  # Use SMOTE for class balancing
        use_class_weights=False  # Additional class weighting
    )
    
    return models, results

if __name__ == "__main__":
    models, evaluation_results = main() 