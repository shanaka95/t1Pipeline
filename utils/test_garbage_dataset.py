#!/usr/bin/env python3
"""
Create a garbage dataset to test if predictors are learning meaningful patterns.
This script:
1. Loads the original top5 dataset
2. Keeps patient IDs and depression results unchanged
3. Randomizes only the cluster-related features
4. Saves as garbage dataset
5. Runs predictors on both original and garbage datasets
"""

import pandas as pd
import numpy as np
import os
import subprocess
import sys
from datetime import datetime

def create_garbage_dataset(input_file, output_file):
    """
    Create a garbage dataset by randomizing cluster features while keeping
    patient IDs and depression results intact.
    """
    print(f"Loading original dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Identify columns to keep unchanged (patient info and depression results)
    keep_unchanged = ['Patient_ID', 'Depression_Binary', 'Depression_3Class', 'Binary_Depression']
    
    # Identify cluster-related columns to randomize
    cluster_cols = []
    for col in df.columns:
        if col.startswith('cluster_') or col in ['total_cluster_activity', 'most_active_cluster', 
                                               'num_active_clusters', 'cluster_diversity']:
            cluster_cols.append(col)
    
    print(f"Keeping unchanged: {keep_unchanged}")
    print(f"Randomizing {len(cluster_cols)} cluster-related columns")
    
    # Create copy of dataframe
    garbage_df = df.copy()
    
    # Randomize each cluster column independently
    np.random.seed(42)  # For reproducibility
    for col in cluster_cols:
        if col != 'most_active_cluster':  # This is categorical, handle differently
            # Shuffle the values for this column
            garbage_df[col] = np.random.permutation(df[col].values)
        else:
            # For most_active_cluster, shuffle the categorical values
            garbage_df[col] = np.random.permutation(df[col].values)
    
    print(f"Saving garbage dataset to: {output_file}")
    garbage_df.to_csv(output_file, index=False)
    
    # Verify the shuffle worked
    print("\nVerification - showing first few rows comparison:")
    print("Original vs Garbage for cluster_000:")
    print(f"Original: {df['cluster_000'].head().tolist()}")
    print(f"Garbage:  {garbage_df['cluster_000'].head().tolist()}")
    
    print(f"\nPatient IDs unchanged: {(df['Patient_ID'] == garbage_df['Patient_ID']).all()}")
    print(f"Depression results unchanged: {(df['Depression_Binary'] == garbage_df['Depression_Binary']).all()}")
    
    return output_file

def run_predictor_workflow(data_file, output_suffix=""):
    """
    Run the top5 workflow on a dataset
    """
    print(f"\n{'='*60}")
    print(f"Running predictors on: {data_file}")
    print(f"{'='*60}")
    
    # Change to predictors directory and import the workflow
    original_dir = os.getcwd()
    try:
        os.chdir('predictors')
        
        # Import the workflow class
        sys.path.insert(0, '.')
        from top5_workflow import DepressionPredictionWorkflow
        
        # Initialize workflow with our specific dataset
        dataset_path = os.path.join('..', data_file)
        print(f"Using dataset: {dataset_path}")
        
        workflow = DepressionPredictionWorkflow(
            processed_data_path=dataset_path,
            feature_info_path='../processed_data/feature_info.pkl'
        )
        
        # Run the complete workflow
        print("Starting workflow execution...")
        models, evaluation = workflow.run_complete_workflow(
            tune_hyperparameters=True,
            balance_method='smote',
            use_class_weights=True,
            include_comparison=True
        )
        
        print(f"Workflow completed successfully!")
        print(f"Trained {len(models)} models")
        
        return True, models, evaluation
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None
            
    finally:
        os.chdir(original_dir)
        if '.' in sys.path:
            sys.path.remove('.')

def main():
    # File paths
    original_file = "processed_data/depression_processed_top5.csv"
    garbage_file = "processed_data/depression_processed_top5_garbage.csv"
    
    print("GARBAGE DATASET TEST")
    print("="*60)
    print("This test creates a 'garbage' dataset where cluster features are randomized")
    print("while keeping patient IDs and depression results unchanged.")
    print("If the model performance drops significantly on garbage data,")
    print("it suggests the model is learning meaningful patterns from cluster features.")
    print("="*60)
    
    # Step 1: Create garbage dataset
    try:
        create_garbage_dataset(original_file, garbage_file)
    except Exception as e:
        print(f"Error creating garbage dataset: {e}")
        return
    
    # Step 2: Run predictors on original dataset
    print(f"\n\nSTEP 1: Running predictors on ORIGINAL dataset")
    success_original, models_original, eval_original = run_predictor_workflow(original_file, "_original")
    
    # Step 3: Run predictors on garbage dataset  
    print(f"\n\nSTEP 2: Running predictors on GARBAGE dataset")
    success_garbage, models_garbage, eval_garbage = run_predictor_workflow(garbage_file, "_garbage")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Original dataset results: {'SUCCESS' if success_original else 'FAILED'}")
    print(f"Garbage dataset results:  {'SUCCESS' if success_garbage else 'FAILED'}")
    
    if success_original and success_garbage:
        print("\nBoth tests completed successfully!")
        print("\nComparing key performance metrics:")
        
        # Extract key metrics for comparison
        try:
            if eval_original and eval_garbage:
                print("\nACCURACY COMPARISON:")
                print("-" * 40)
                for model_name in eval_original.keys():
                    if model_name in eval_garbage:
                        orig_acc = eval_original[model_name].get('test_accuracy', 'N/A')
                        garb_acc = eval_garbage[model_name].get('test_accuracy', 'N/A')
                        print(f"{model_name:20}: Original={orig_acc:.3f}, Garbage={garb_acc:.3f}")
                        
                print("\nF1 SCORE COMPARISON:")
                print("-" * 40)
                for model_name in eval_original.keys():
                    if model_name in eval_garbage:
                        orig_f1 = eval_original[model_name].get('test_f1', 'N/A')
                        garb_f1 = eval_garbage[model_name].get('test_f1', 'N/A')
                        print(f"{model_name:20}: Original={orig_f1:.3f}, Garbage={garb_f1:.3f}")
                        
                print("\nRESULT INTERPRETATION:")
                print("-" * 40)
                print("If garbage dataset shows significantly LOWER performance:")
                print("✓ Model is learning meaningful patterns from cluster features")
                print("\nIf garbage dataset shows SIMILAR performance:")
                print("⚠ Model might be overfitting or cluster features may not be informative")
                
        except Exception as e:
            print(f"Error comparing results: {e}")
            print("Check the saved results files for detailed comparison")
            
        print("\nDetailed results saved in:")
        print("- Original: Check saved_models/ directory with timestamp")
        print("- Garbage: Check saved_models/ directory with timestamp")
    else:
        print("\nSome tests failed. Check the error messages above.")
    
    print(f"\nFiles created:")
    print(f"- Garbage dataset: {garbage_file}")
    print(f"- Check saved_models/ directory for result files")

if __name__ == "__main__":
    main()