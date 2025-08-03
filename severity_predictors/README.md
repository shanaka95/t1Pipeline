# Depression Severity Prediction Workflows

This directory contains comprehensive workflows for predicting depression severity using both Top1 and Top5 clustering approaches, with SMOTE-based class imbalance handling.

## Overview

### 🎯 **Purpose**
Predict depression severity levels (Mild/Subclinical, Moderate, Severe) using behavioral features extracted from video data, comparing different clustering granularities and balance methods.

### 📊 **Approaches**
- **Top5 Clustering**: Uses 100 fine-grained action clusters as features
- **Top1 Action Classes**: Uses 52 broader action categories as features
- **SMOTE Comparison**: Systematic evaluation with and without synthetic oversampling

## File Structure

### Core Workflow Scripts
- **`top5_workflow_with_smote.py`** - Complete Top5 severity prediction pipeline
- **`top1_workflow_with_smote.py`** - Complete Top1 severity prediction pipeline
- **`workflow_with_smote.py`** - Original general-purpose workflow

### Model Classes
- **`base_severity_model.py`** - Base class with common functionality for both approaches
- **`xgb_severity_model.py`** - XGBoost implementation
- **`random_forest_severity_model.py`** - Random Forest implementation
- **`svm_severity_model.py`** - Support Vector Machine implementation

### Testing & Analysis
- **`test_workflows.py`** - Comprehensive test suite for both workflows
- **`APPROACH_ANALYSIS.md`** - Detailed analysis of approach correctness

## Quick Start

### 1. **Test the Workflows**
```bash
# Activate virtual environment
source ../env/bin/activate

# Run comprehensive tests
python test_workflows.py
```

### 2. **Run Top5 Workflow**
```bash
python top5_workflow_with_smote.py
```

### 3. **Run Top1 Workflow**
```bash
python top1_workflow_with_smote.py
```

## Data Requirements

### Top5 Clustering Data
- **Data File**: `../processed_data/depression_processed_top5.csv`
- **Features**: 100 cluster columns (cluster_000 to cluster_099) + derived features
- **Feature Info**: `../processed_data/feature_info.pkl`

### Top1 Action Class Data
- **Data File**: `../processed_data/depression_processed_top1.csv`
- **Features**: 52 action class columns + engineered features
- **Feature Info**: `../processed_data/top1_feature_info.pkl`

## Key Features

### 🔄 **Class Imbalance Handling**
- **SMOTE Variants**: Standard SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek, SMOTEENN
- **Systematic Comparison**: Models trained with and without SMOTE
- **Performance Analysis**: Detailed impact assessment

### 🤖 **Model Ensemble**
- **XGBoost**: Gradient boosting with multi-class support
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel

### 📈 **Comprehensive Evaluation**
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Cross-validation**: Stratified train-test splits
- **Visualizations**: Performance comparisons, confusion matrices

### 🛡️ **Data Integrity**
- **Target Leakage Prevention**: Automatic filtering of target-related features
- **Feature Validation**: Ensures only appropriate behavioral features are used
- **Missing Value Handling**: Median imputation for numerical features

## Test Results Summary

✅ **All Tests Passed (6/6)**
- ✅ Data Availability: All required files present
- ✅ Top5 Basic Functionality: Workflow initializes and loads data correctly
- ✅ Top1 Basic Functionality: Workflow initializes and loads data correctly  
- ✅ Base Model Compatibility: Handles both feature structures
- ✅ Feature Integrity: No target leakage detected
- ✅ Quick Training: Models train successfully

## Expected Outputs

### Models
- **Location**: `../saved_models/top5_severity/` and `../saved_models/top1_severity/`
- **Files**: Trained model objects (.pkl files) with timestamps

### Visualizations
- **Location**: `../severity_results/top5/` and `../severity_results/top1/`
- **Contents**: 
  - SMOTE comparison plots
  - Performance analysis charts
  - Confusion matrices

### Results
- **Comparison CSV**: SMOTE vs no-SMOTE performance metrics
- **Summary JSON**: Workflow configuration and best model info

## Class Distribution Analysis

### Top5 Clustering (221 samples)
- **Mild/Subclinical**: 131 (59.3%)
- **Moderate**: 77 (34.8%)
- **Severe**: 13 (5.9%)
- **Imbalance Ratio**: 10.08 → **SMOTE beneficial**

### Top1 Action Classes (226 samples)
- **Class 0**: 63 (27.9%)
- **Mild/Subclinical**: 70 (31.0%)
- **Moderate**: 93 (41.2%)
- **Imbalance Ratio**: 1.48 → **Moderate imbalance**

## Performance Expectations

### Initial Quick Training Results
- **Top5 (without SMOTE)**: 56.7% accuracy, 52.6% F1-score
- **Top1 (without SMOTE)**: 35.3% accuracy, 34.5% F1-score

*Note: These are preliminary results without hyperparameter tuning or SMOTE. Full workflow runs will provide comprehensive comparisons.*

## Usage Examples

### Run Full Top5 Analysis
```python
from top5_workflow_with_smote import Top5SeverityWorkflowWithSMOTE

# Initialize and run complete workflow
workflow = Top5SeverityWorkflowWithSMOTE()
models, results, improvements = workflow.run_complete_workflow()

# Access best model
best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
print(f"Best Top5 model: {best_model_name}")
```

### Run Full Top1 Analysis
```python
from top1_workflow_with_smote import Top1SeverityWorkflowWithSMOTE

# Initialize and run complete workflow
workflow = Top1SeverityWorkflowWithSMOTE()
models, results, improvements = workflow.run_complete_workflow()

# Access best model
best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
print(f"Best Top1 model: {best_model_name}")
```

## Approach Validation

### ✅ **Strengths**
- Sound machine learning pipeline with proper validation
- Comprehensive class imbalance handling
- Multiple algorithm comparison
- Feature integrity verification
- Systematic SMOTE impact analysis

### ⚠️ **Considerations**
- **Clinical Validation**: Severity labels should be validated against clinical standards
- **Temporal Modeling**: Could benefit from sequential behavior patterns
- **Generalizability**: Performance on new populations needs validation
- **Interpretability**: Feature importance analysis for clinical actionability

## Next Steps

1. **Run Full Workflows**: Execute both Top1 and Top5 complete pipelines
2. **Compare Results**: Analyze which approach performs better
3. **Clinical Validation**: Correlate predictions with clinical assessment scores
4. **Feature Analysis**: Examine which behavioral patterns predict severity
5. **Deployment**: Package best-performing model for clinical use

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.4.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Contact & Support

For technical issues or questions about the severity prediction approach, refer to the comprehensive analysis in `APPROACH_ANALYSIS.md` or review the test results from `test_workflows.py`.