#!/usr/bin/env python3
"""
Test script for Top1 and Top5 severity prediction workflows
This script performs basic functionality tests to ensure the workflows work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_availability():
    """Test if required data files are available"""
    print("="*60)
    print("TESTING DATA AVAILABILITY")
    print("="*60)
    
    # Check Top5 data
    top5_data_path = '../processed_data/depression_processed_top5.csv'
    top5_feature_path = '../processed_data/feature_info.pkl'
    
    print(f"\nTop5 Data Files:")
    print(f"  - Data file: {top5_data_path} - {'✓' if os.path.exists(top5_data_path) else '✗'}")
    print(f"  - Feature file: {top5_feature_path} - {'✓' if os.path.exists(top5_feature_path) else '✗'}")
    
    # Check Top1 data
    top1_data_path = '../processed_data/depression_processed_top1.csv'
    top1_feature_path = '../processed_data/top1_feature_info.pkl'
    
    print(f"\nTop1 Data Files:")
    print(f"  - Data file: {top1_data_path} - {'✓' if os.path.exists(top1_data_path) else '✗'}")
    print(f"  - Feature file: {top1_feature_path} - {'✓' if os.path.exists(top1_feature_path) else '✗'}")
    
    return (os.path.exists(top5_data_path) and os.path.exists(top5_feature_path) and
            os.path.exists(top1_data_path) and os.path.exists(top1_feature_path))

def test_top5_workflow_basic():
    """Test basic functionality of Top5 workflow"""
    print("\n" + "="*60)
    print("TESTING TOP5 WORKFLOW - BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        from top5_workflow_with_smote import Top5SeverityWorkflowWithSMOTE
        
        # Initialize workflow
        print("Initializing Top5 workflow...")
        workflow = Top5SeverityWorkflowWithSMOTE()
        
        # Test data loading
        print("Testing data loading...")
        df, feature_info = workflow.load_data_and_features()
        
        print(f"✓ Top5 data loaded successfully: {df.shape}")
        print(f"✓ Feature info loaded with keys: {list(feature_info.keys())}")
        
        # Check if Depression_3Class exists for severity prediction
        if 'Depression_3Class' in df.columns:
            severity_dist = df['Depression_3Class'].value_counts().sort_index()
            print(f"✓ Severity distribution: {dict(severity_dist)}")
        else:
            print("⚠️ Depression_3Class not found, will use binary target")
        
        return True
        
    except Exception as e:
        print(f"✗ Top5 workflow test failed: {str(e)}")
        return False

def test_top1_workflow_basic():
    """Test basic functionality of Top1 workflow"""
    print("\n" + "="*60)
    print("TESTING TOP1 WORKFLOW - BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        from top1_workflow_with_smote import Top1SeverityWorkflowWithSMOTE
        
        # Initialize workflow
        print("Initializing Top1 workflow...")
        workflow = Top1SeverityWorkflowWithSMOTE()
        
        # Test data loading
        print("Testing data loading...")
        df, feature_info = workflow.load_data_and_features()
        
        print(f"✓ Top1 data loaded successfully: {df.shape}")
        print(f"✓ Feature info loaded with keys: {list(feature_info.keys())}")
        
        # Check targets available
        targets_available = []
        if 'Depression_Binary' in df.columns:
            targets_available.append('binary')
            binary_dist = df['Depression_Binary'].value_counts().sort_index()
            print(f"✓ Binary distribution: {dict(binary_dist)}")
        
        if 'Depression_3Class' in df.columns:
            targets_available.append('3-class')
            severity_dist = df['Depression_3Class'].value_counts().sort_index()
            print(f"✓ Severity distribution: {dict(severity_dist)}")
        
        print(f"✓ Available targets: {targets_available}")
        
        return True
        
    except Exception as e:
        print(f"✗ Top1 workflow test failed: {str(e)}")
        return False

def test_base_model_compatibility():
    """Test if base model works with both Top1 and Top5 data"""
    print("\n" + "="*60)
    print("TESTING BASE MODEL COMPATIBILITY")
    print("="*60)
    
    try:
        from base_severity_model import BaseSeverityModel
        
        # Test with Top5 data
        print("Testing base model with Top5 data...")
        top5_model = BaseSeverityModel(
            '../processed_data/depression_processed_top5.csv',
            '../processed_data/feature_info.pkl'
        )
        top5_model.load_processed_data()
        X_top5, y_3class_top5, y_binary_top5, features_top5 = top5_model.prepare_features_targets()
        print(f"✓ Top5 features prepared: {X_top5.shape}, {len(features_top5)} features")
        
        # Test with Top1 data
        print("Testing base model with Top1 data...")
        top1_model = BaseSeverityModel(
            '../processed_data/depression_processed_top1.csv',
            '../processed_data/top1_feature_info.pkl'
        )
        top1_model.load_processed_data()
        X_top1, y_3class_top1, y_binary_top1, features_top1 = top1_model.prepare_features_targets()
        print(f"✓ Top1 features prepared: {X_top1.shape}, {len(features_top1)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Base model compatibility test failed: {str(e)}")
        return False

def test_feature_integrity():
    """Test feature integrity and ensure no target leakage"""
    print("\n" + "="*60)
    print("TESTING FEATURE INTEGRITY")
    print("="*60)
    
    try:
        from base_severity_model import BaseSeverityModel
        
        # Test Top5 feature integrity
        print("Checking Top5 feature integrity...")
        top5_model = BaseSeverityModel(
            '../processed_data/depression_processed_top5.csv',
            '../processed_data/feature_info.pkl'
        )
        top5_model.load_processed_data()
        X_top5, y_3class_top5, y_binary_top5, features_top5 = top5_model.prepare_features_targets()
        top5_model.verify_feature_integrity(features_top5)
        
        # Check for target leakage in Top5
        target_columns = ['Depression_Binary', 'Depression_3Class', 'Binary_Depression']
        leakage_top5 = [col for col in features_top5 if col in target_columns]
        if leakage_top5:
            print(f"✗ Top5 target leakage detected: {leakage_top5}")
            return False
        else:
            print("✓ Top5 no target leakage detected")
        
        # Test Top1 feature integrity
        print("Checking Top1 feature integrity...")
        top1_model = BaseSeverityModel(
            '../processed_data/depression_processed_top1.csv',
            '../processed_data/top1_feature_info.pkl'
        )
        top1_model.load_processed_data()
        X_top1, y_3class_top1, y_binary_top1, features_top1 = top1_model.prepare_features_targets()
        top1_model.verify_feature_integrity(features_top1)
        
        # Check for target leakage in Top1
        leakage_top1 = [col for col in features_top1 if col in target_columns]
        if leakage_top1:
            print(f"✗ Top1 target leakage detected: {leakage_top1}")
            return False
        else:
            print("✓ Top1 no target leakage detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature integrity test failed: {str(e)}")
        return False

def test_quick_model_training():
    """Test quick model training with minimal configuration"""
    print("\n" + "="*60)
    print("TESTING QUICK MODEL TRAINING")
    print("="*60)
    
    try:
        from base_severity_model import BaseSeverityModel
        from xgb_severity_model import XGBoostSeverityModel
        
        # Test with Top5 data - quick training
        print("Testing quick Top5 model training...")
        top5_trainer = XGBoostSeverityModel(
            '../processed_data/depression_processed_top5.csv',
            '../processed_data/feature_info.pkl'
        )
        top5_trainer.load_processed_data()
        X, y_3class, y_binary, feature_cols = top5_trainer.prepare_features_targets()
        X_train, X_test, y_train, y_test = top5_trainer.split_data(X, y_3class, test_size=0.3)
        
        # Quick train without hyperparameter tuning
        model = top5_trainer.train_xgboost_model(
            X_train, y_train, X_test, y_test,
            model_name="test_top5_quick",
            tune_hyperparameters=False,
            balance_method='none'
        )
        print("✓ Top5 quick training completed successfully")
        
        # Test with Top1 data - quick training
        print("Testing quick Top1 model training...")
        top1_trainer = XGBoostSeverityModel(
            '../processed_data/depression_processed_top1.csv',
            '../processed_data/top1_feature_info.pkl'
        )
        top1_trainer.load_processed_data()
        X, y_3class, y_binary, feature_cols = top1_trainer.prepare_features_targets()
        
        # Use appropriate target
        target = y_3class if y_3class is not None else y_binary
        X_train, X_test, y_train, y_test = top1_trainer.split_data(X, target, test_size=0.3)
        
        # Quick train without hyperparameter tuning
        model = top1_trainer.train_xgboost_model(
            X_train, y_train, X_test, y_test,
            model_name="test_top1_quick",
            tune_hyperparameters=False,
            balance_method='none'
        )
        print("✓ Top1 quick training completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick model training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("SEVERITY PREDICTION WORKFLOWS - COMPREHENSIVE TESTING")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    test_results = {}
    
    # Run tests
    test_results['data_availability'] = test_data_availability()
    test_results['top5_basic'] = test_top5_workflow_basic()
    test_results['top1_basic'] = test_top1_workflow_basic()
    test_results['base_model_compatibility'] = test_base_model_compatibility()
    test_results['feature_integrity'] = test_feature_integrity()
    test_results['quick_training'] = test_quick_model_training()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Workflows are ready for use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)