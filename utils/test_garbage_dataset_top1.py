#!/usr/bin/env python3
"""
Create a garbage dataset to test if predictors are learning meaningful patterns for TOP1 dataset.
This script:
1. Loads the original top1 dataset
2. Keeps patient IDs and depression results unchanged
3. Randomizes only the action class-related features
4. Saves as garbage dataset
5. Runs predictors on both original and garbage datasets
"""

import pandas as pd
import numpy as np
import os
import subprocess
import sys
from datetime import datetime

def create_garbage_dataset_top1(input_file, output_file):
    """
    Create a garbage dataset by randomizing action class features while keeping
    patient IDs and depression results intact.
    """
    print(f"Loading original top1 dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Identify columns to keep unchanged (patient info and depression results)
    keep_unchanged = ['Patient_ID', 'Depression_Binary', 'Depression_3Class', 'depressed']
    
    # Identify action class-related columns to randomize
    action_cols = []
    for col in df.columns:
        if col.startswith('action_class_') or col in ['total_action_activity', 'most_active_action', 
                                                    'num_active_actions', 'action_diversity']:
            action_cols.append(col)
    
    print(f"Keeping unchanged: {keep_unchanged}")
    print(f"Randomizing {len(action_cols)} action class-related columns")
    
    # Create copy of dataframe
    garbage_df = df.copy()
    
    # Randomize each action column independently
    np.random.seed(42)  # For reproducibility
    for col in action_cols:
        if col != 'most_active_action':  # This is categorical, handle differently
            # Shuffle the values for this column
            garbage_df[col] = np.random.permutation(df[col].values)
        else:
            # For most_active_action, shuffle the categorical values
            garbage_df[col] = np.random.permutation(df[col].values)
    
    print(f"Saving garbage dataset to: {output_file}")
    garbage_df.to_csv(output_file, index=False)
    
    # Verify the shuffle worked
    print("\nVerification - showing first few rows comparison:")
    print("Original vs Garbage for action_class_00:")
    print(f"Original: {df['action_class_00'].head().tolist()}")
    print(f"Garbage:  {garbage_df['action_class_00'].head().tolist()}")
    
    print(f"\nPatient IDs unchanged: {(df['Patient_ID'] == garbage_df['Patient_ID']).all()}")
    print(f"Depression results unchanged: {(df['Depression_Binary'] == garbage_df['Depression_Binary']).all()}")
    
    return output_file

def run_predictor_workflow_top1(data_file, output_suffix=""):
    """
    Run the top1 workflow on a dataset
    """
    print(f"\n{'='*60}")
    print(f"Running TOP1 predictors on: {data_file}")
    print(f"{'='*60}")
    
    # Change to predictors directory and import the workflow
    original_dir = os.getcwd()
    try:
        os.chdir('predictors')
        
        # Import the workflow class
        sys.path.insert(0, '.')
        from top1_workflow_comprehensive import ComprehensiveTop1DepressionPredictionWorkflow
        
        # Initialize workflow with our specific dataset
        dataset_path = os.path.join('..', data_file)
        print(f"Using dataset: {dataset_path}")
        
        workflow = ComprehensiveTop1DepressionPredictionWorkflow(
            dataset_path=dataset_path
        )
        
        # Run the complete workflow
        print("Starting TOP1 workflow execution...")
        models, evaluation = workflow.run_complete_workflow(
            include_comparison=True
        )
        
        print(f"TOP1 Workflow completed successfully!")
        print(f"Trained {len(models)} models")
        
        return True, models, evaluation
        
    except Exception as e:
        print(f"Error running TOP1 workflow: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None
            
    finally:
        os.chdir(original_dir)
        if '.' in sys.path:
            sys.path.remove('.')

def main():
    # File paths
    original_file = "processed_data/depression_processed_top1.csv"
    garbage_file = "processed_data/depression_processed_top1_garbage.csv"
    
    print("GARBAGE DATASET TEST - TOP1 ACTION CLASSES")
    print("="*60)
    print("This test creates a 'garbage' dataset where action class features are randomized")
    print("while keeping patient IDs and depression results unchanged.")
    print("If the model performance drops significantly on garbage data,")
    print("it suggests the model is learning meaningful patterns from action class features.")
    print("="*60)
    
    # Step 1: Create garbage dataset
    try:
        create_garbage_dataset_top1(original_file, garbage_file)
    except Exception as e:
        print(f"Error creating garbage dataset: {e}")
        return
    
    # Step 2: Run predictors on original dataset
    print(f"\n\nSTEP 1: Running TOP1 predictors on ORIGINAL dataset")
    success_original, models_original, eval_original = run_predictor_workflow_top1(original_file, "_original")
    
    # Step 3: Run predictors on garbage dataset  
    print(f"\n\nSTEP 2: Running TOP1 predictors on GARBAGE dataset")
    success_garbage, models_garbage, eval_garbage = run_predictor_workflow_top1(garbage_file, "_garbage")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("TOP1 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Original dataset results: {'SUCCESS' if success_original else 'FAILED'}")
    print(f"Garbage dataset results:  {'SUCCESS' if success_garbage else 'FAILED'}")
    
    if success_original and success_garbage:
        print("\nBoth TOP1 tests completed successfully!")
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
                        if isinstance(orig_acc, (int, float)) and isinstance(garb_acc, (int, float)):
                            print(f"{model_name:20}: Original={orig_acc:.3f}, Garbage={garb_acc:.3f}")
                        else:
                            print(f"{model_name:20}: Original={orig_acc}, Garbage={garb_acc}")
                        
                print("\nF1 SCORE COMPARISON:")
                print("-" * 40)
                for model_name in eval_original.keys():
                    if model_name in eval_garbage:
                        orig_f1 = eval_original[model_name].get('test_f1', 'N/A')
                        garb_f1 = eval_garbage[model_name].get('test_f1', 'N/A')
                        if isinstance(orig_f1, (int, float)) and isinstance(garb_f1, (int, float)):
                            print(f"{model_name:20}: Original={orig_f1:.3f}, Garbage={garb_f1:.3f}")
                        else:
                            print(f"{model_name:20}: Original={orig_f1}, Garbage={garb_f1}")
                        
                print("\nRESULT INTERPRETATION:")
                print("-" * 40)
                print("If garbage dataset shows significantly LOWER performance:")
                print("✓ Model is learning meaningful patterns from action class features")
                print("\nIf garbage dataset shows SIMILAR performance:")
                print("⚠ Model might be overfitting or action features may not be informative")
                
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