"""
Random Forest Model for Depression Prediction
This module provides Random Forest-specific functionality for depression prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from base_model import BaseDepressionModel

class RandomForestDepressionModel(BaseDepressionModel):
    def __init__(self, processed_data_path='../processed_data/depression_processed.csv',
                 feature_info_path='../processed_data/feature_info.pkl'):
        """Initialize the Random Forest model trainer"""
        super().__init__(processed_data_path, feature_info_path)
        
    def train_random_forest_model(self, X_train, y_train, X_test, y_test, 
                                model_name="random_forest", tune_hyperparameters=True):
        """Train Random Forest model with optional hyperparameter tuning"""
        print(f"\n🚀 Training Random Forest model: {model_name}")
        
        if tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Create Random Forest classifier
            rf_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                rf_model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=1, return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters with some optimization
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
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
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest model"""
        if 'random_forest' not in self.models:
            return
            
        model = self.models['random_forest']
        feature_names = self.results['random_forest']['feature_names']
        
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
        plt.title('Random Forest Feature Importance (Top 20)')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('../model_results/rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance to CSV
        importance_df.to_csv('../model_results/rf_feature_importance.csv', index=False)
        print("✅ Random Forest feature importance plot and CSV saved")
    
    def run_training_pipeline(self, tune_hyperparameters=True):
        """Run the complete Random Forest training pipeline"""
        print("🚀 Starting Random Forest Training Pipeline for Depression Prediction")
        print("="*70)
        
        # Load data
        self.load_processed_data()
        
        # Prepare features and targets
        X, y_binary, y_3class, feature_cols = self.prepare_features_targets()
        
        # Verify feature integrity
        self.verify_feature_integrity(feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_binary)
        
        # Train Random Forest model
        rf_model = self.train_random_forest_model(
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
        
        print("\n🎉 Random Forest TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Models trained: {len(self.models)}")
        print(f"🏆 Best model by AUC: {max(evaluation_results.items(), key=lambda x: x[1]['auc_roc'])}")
        print(f"💾 Models saved to 'saved_models/' directory")
        print(f"📊 Results saved to 'model_results/' directory")
        
        return self.models, evaluation_results

def main():
    """Main function to run Random Forest training"""
    # Initialize trainer
    trainer = RandomForestDepressionModel()
    
    # Run complete pipeline
    models, results = trainer.run_training_pipeline(tune_hyperparameters=True)
    
    return models, results

if __name__ == "__main__":
    models, evaluation_results = main() 