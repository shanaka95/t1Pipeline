"""
Simplified Test Workflow for Depression Severity Prediction
This script tests all three models for severity prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
import xgboost as xgb

warnings.filterwarnings('ignore')

def load_data_and_prepare_features():
    """Load data and prepare features for severity prediction"""
    print("Loading data and preparing features...")
    
    # Load data
    df = pd.read_csv('../processed_data/depression_processed.csv')
    
    # Load feature info
    with open('../processed_data/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    
    print(f"Data loaded: {df.shape}")
    
    # Prepare features (scaled cluster features + derived)
    feature_cols = feature_info['scaled_cluster_columns'].copy()
    derived_features = [f for f in feature_info['derived_columns'] if f != 'most_active_cluster']
    feature_cols.extend(derived_features)
    
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['Depression_3Class']
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution: {dict(y.value_counts().sort_index())}")
    
    return X, y, feature_cols, df

def run_eda():
    """Run basic EDA for severity prediction"""
    print("\n" + "="*60)
    print("RUNNING EDA FOR SEVERITY PREDICTION")
    print("="*60)
    
    df = pd.read_csv('../processed_data/depression_processed.csv')
    
    # Severity distribution
    severity_counts = df['Depression_3Class'].value_counts().sort_index()
    severity_labels = {1: 'Mild/Subclinical', 2: 'Moderate', 3: 'Severe'}
    
    print(f"\nSeverity Distribution:")
    for cls in severity_counts.index:
        label = severity_labels.get(cls, f'Class {cls}')
        percentage = (severity_counts[cls] / len(df)) * 100
        print(f"  {label} (Class {cls}): {severity_counts[cls]} ({percentage:.1f}%)")
    
    # Class imbalance analysis
    class_balance = severity_counts.max() / severity_counts.min()
    print(f"\nClass imbalance ratio: {class_balance:.2f}")
    if class_balance > 3:
        print("WARNING: Significant class imbalance detected")
    else:
        print("Class distribution is reasonably balanced")
    
    # Create visualization directory
    os.makedirs('../severity_visualizations', exist_ok=True)
    
    # Plot severity distribution
    plt.figure(figsize=(10, 6))
    severity_data = severity_counts.values
    labels = [severity_labels[i] for i in severity_counts.index]
    colors = ['lightblue', 'orange', 'red']
    
    plt.subplot(1, 2, 1)
    plt.pie(severity_data, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('Depression Severity Distribution')
    
    plt.subplot(1, 2, 2)
    plt.bar(labels, severity_data, color=colors, alpha=0.7)
    plt.title('Depression Severity Counts')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../severity_visualizations/severity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EDA visualizations saved to '../severity_visualizations/' directory")
    return df

def train_all_models(X_train, y_train, X_test, y_test):
    """Train all three models for severity prediction"""
    print("\n" + "="*60)
    print("TRAINING ALL MODELS FOR SEVERITY PREDICTION")
    print("="*60)
    
    models = {}
    results = {}
    
    # 1. XGBoost
    print(f"\n{'='*20} XGBoost {'='*20}")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'y_test': y_test,
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_pred_proba
    }
    
    accuracy = accuracy_score(y_test, xgb_pred)
    f1 = f1_score(y_test, xgb_pred, average='weighted')
    print(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # 2. Random Forest
    print(f"\n{'='*20} Random Forest {'='*20}")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'y_test': y_test,
        'y_pred': rf_pred,
        'y_pred_proba': rf_pred_proba
    }
    
    accuracy = accuracy_score(y_test, rf_pred)
    f1 = f1_score(y_test, rf_pred, average='weighted')
    print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # 3. SVM
    print(f"\n{'='*20} SVM {'='*20}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_pred_proba = svm_model.predict_proba(X_test_scaled)
    
    models['SVM'] = svm_model
    results['SVM'] = {
        'y_test': y_test,
        'y_pred': svm_pred,
        'y_pred_proba': svm_pred_proba
    }
    
    accuracy = accuracy_score(y_test, svm_pred)
    f1 = f1_score(y_test, svm_pred, average='weighted')
    print(f"SVM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return models, results

def evaluate_models(results):
    """Evaluate all models comprehensively"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    evaluation_results = {}
    
    for model_name, result in results.items():
        y_test = result['y_test']
        y_pred = result['y_pred']
        y_pred_proba = result['y_pred_proba']
        
        print(f"\n{model_name.upper()} Results:")
        print("-" * 40)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        severity_labels = ['Mild/Subclinical', 'Moderate', 'Severe']
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=severity_labels))
    
    return evaluation_results

def create_visualizations(results):
    """Create visualizations for model comparison"""
    print("\nCreating visualizations...")
    
    # Create output directory
    os.makedirs('../severity_results', exist_ok=True)
    
    # 1. Model Comparison Plot
    metrics_data = []
    for model_name, result in results.items():
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics_data.append({'Model': model_name, 'Metric': 'Accuracy', 'Score': accuracy})
        metrics_data.append({'Model': model_name, 'Metric': 'F1-Score', 'Score': f1})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Metric', y='Score', hue='Model')
    plt.title('Severity Prediction Model Comparison')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('../severity_results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrices
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, result) in enumerate(results.items()):
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                   xticklabels=['Mild', 'Moderate', 'Severe'],
                   yticklabels=['Mild', 'Moderate', 'Severe'])
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Severity')
        axes[i].set_ylabel('Actual Severity')
    
    plt.tight_layout()
    plt.savefig('../severity_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to '../severity_results/' directory")

def main():
    """Main function to run the complete severity prediction workflow"""
    print("Starting Comprehensive Depression Severity Prediction Test")
    print("="*70)
    
    # Step 1: Run EDA
    df = run_eda()
    
    # Step 2: Load and prepare data
    X, y, feature_cols, df = load_data_and_prepare_features()
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Step 4: Train all models
    models, results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate models
    evaluation_results = evaluate_models(results)
    
    # Step 6: Create visualizations
    create_visualizations(results)
    
    # Step 7: Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\nModels Trained: {len(models)}")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    print(f"\nPerformance Summary:")
    best_model = max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])
    print(f"  Best Model: {best_model[0]} (F1: {best_model[1]['f1_score']:.4f})")
    
    print(f"\nDetailed Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"  {model_name}:")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - F1-Score: {metrics['f1_score']:.4f}")
    
    print(f"\nOutput Files:")
    print(f"  - Visualizations: ../severity_results/ directory")
    print(f"  - EDA Visualizations: ../severity_visualizations/ directory")
    
    print(f"\nSEVERITY PREDICTION WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return models, results, evaluation_results

if __name__ == "__main__":
    models, results, evaluation = main() 