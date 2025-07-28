"""
Logistic Regression Model for Depression Prediction
This module provides Logistic Regression-specific functionality for depression prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from base_model import BaseDepressionModel

class LogisticRegressionDepressionModel(BaseDepressionModel):
    def __init__(self, processed_data_path='../processed_data/depression_processed.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the Logistic Regression model trainer"""
        super().__init__(processed_data_path, feature_info_path)
        self.scaler = StandardScaler()
        
    def train_logistic_regression_model(self, X_train, y_train, X_test, y_test, 
                                      model_name="logistic_regression", tune_hyperparameters=True, 
                                      balance_method='none', use_class_weights=False):
        """Train Logistic Regression model with optional hyperparameter tuning and class balancing"""
        print(f"\n🚀 Training Logistic Regression model: {model_name}")
        
        # Handle class imbalance
        if balance_method != 'none':
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train, y_train, method=balance_method
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features for Logistic Regression
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Set class weights
        class_weight = 'balanced' if use_class_weights else None
        if use_class_weights:
            print(f"Using class_weight: {class_weight}")
        
        if tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            
            # Define parameter grid (reduced for faster training)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
            
            # Create Logistic Regression classifier with class weights
            lr_model = LogisticRegression(
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                lr_model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=1, return_train_score=True
            )
            
            grid_search.fit(X_train_scaled, y_train_balanced)
            
            # Best model
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters with optimization
            best_model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            best_model.fit(X_train_scaled, y_train_balanced)
        
        # Make predictions on original test set
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
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
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Model trained successfully!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        
        return best_model
    
    def plot_feature_importance(self):
        """Plot feature importance for Logistic Regression model"""
        if 'logistic_regression' not in self.models:
            return
            
        model = self.models['logistic_regression']
        feature_names = self.results['logistic_regression']['feature_names']
        
        # Get feature importance (coefficients)
        importance = np.abs(model.coef_[0])
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(20)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('Logistic Regression Feature Importance (Top 20)')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('../model_results/lr_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance to CSV
        importance_df.to_csv('../model_results/lr_feature_importance.csv', index=False)
        print("Logistic Regression feature importance plot and CSV saved")
    
    def run_training_pipeline(self, tune_hyperparameters=True):
        """Run the complete Logistic Regression training pipeline"""
        print("🚀 Starting Logistic Regression Training Pipeline for Depression Prediction")
        print("="*70)
        
        # Load data
        self.load_processed_data()
        
        # Prepare features and targets
        X, y_binary, y_3class, feature_cols = self.prepare_features_targets()
        
        # Verify feature integrity
        self.verify_feature_integrity(feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_binary)
        
        # Train Logistic Regression model
        lr_model = self.train_logistic_regression_model(
            X_train, y_train, X_test, y_test, 
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Save models
        self.save_models()
        
        print("\nLogistic Regression TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Models trained: {len(self.models)}")
        print(f"Best model by AUC: {max(evaluation_results.items(), key=lambda x: x[1]['auc_roc'])}")
        print(f"Models saved to 'saved_models/' directory")
        print(f"Results saved to 'model_results/' directory")
        
        return self.models, evaluation_results

def main():
    """Main function to run Logistic Regression training"""
    # Initialize trainer
    trainer = LogisticRegressionDepressionModel()
    
    # Run complete pipeline
    models, results = trainer.run_training_pipeline(tune_hyperparameters=True)
    
    return models, results

if __name__ == "__main__":
    models, evaluation_results = main() 