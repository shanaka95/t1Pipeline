#!/usr/bin/env python3
"""
Comprehensive Summary of Garbage Dataset Tests
Compare performance on original vs randomized features for both TOP5 and TOP1 datasets
"""

import pandas as pd

def load_and_compare_results():
    print("=" * 80)
    print("COMPREHENSIVE GARBAGE DATASET TEST RESULTS SUMMARY")
    print("=" * 80)
    print()
    print("This test randomizes cluster/action features while keeping patient IDs and")
    print("depression results unchanged to test if models learn meaningful patterns.")
    print()
    
    # TOP5 Results
    print("🔬 TOP5 CLUSTER FEATURES TEST")
    print("-" * 50)
    
    try:
        # Load TOP5 results
        top5_original = pd.read_csv('saved_models/comprehensive_evaluation_20250803_142607.csv', index_col=0)
        top5_garbage = pd.read_csv('saved_models/comprehensive_evaluation_20250803_142632.csv', index_col=0)
        
        print("Original TOP5 Performance:")
        for model in top5_original.index:
            acc = top5_original.loc[model, 'accuracy']
            f1 = top5_original.loc[model, 'f1_score']
            auc = top5_original.loc[model, 'auc_roc']
            print(f"  {model:20}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        print("\nGarbage TOP5 Performance:")
        for model in top5_garbage.index:
            acc = top5_garbage.loc[model, 'accuracy']
            f1 = top5_garbage.loc[model, 'f1_score']
            auc = top5_garbage.loc[model, 'auc_roc']
            print(f"  {model:20}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            
        print("\nTOP5 Performance Differences (Original - Garbage):")
        for model in top5_original.index:
            acc_diff = top5_original.loc[model, 'accuracy'] - top5_garbage.loc[model, 'accuracy']
            f1_diff = top5_original.loc[model, 'f1_score'] - top5_garbage.loc[model, 'f1_score']
            auc_diff = top5_original.loc[model, 'auc_roc'] - top5_garbage.loc[model, 'auc_roc']
            print(f"  {model:20}: Δ Accuracy={acc_diff:+.3f}, Δ F1={f1_diff:+.3f}, Δ AUC={auc_diff:+.3f}")
            
    except Exception as e:
        print(f"Error loading TOP5 results: {e}")
    
    print()
    
    # TOP1 Results
    print("🎯 TOP1 ACTION CLASS FEATURES TEST")
    print("-" * 50)
    
    try:
        # Load TOP1 results
        top1_original = pd.read_csv('top1_comprehensive_results_20250803_142856/comprehensive_evaluation_20250803_142856.csv', index_col=0)
        top1_garbage = pd.read_csv('top1_comprehensive_results_20250803_142900/comprehensive_evaluation_20250803_142900.csv', index_col=0)
        
        print("Original TOP1 Performance:")
        for model in top1_original.index:
            acc = top1_original.loc[model, 'accuracy']
            f1 = top1_original.loc[model, 'f1_score']
            auc = top1_original.loc[model, 'auc_roc']
            print(f"  {model:20}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        print("\nGarbage TOP1 Performance:")
        for model in top1_garbage.index:
            acc = top1_garbage.loc[model, 'accuracy']
            f1 = top1_garbage.loc[model, 'f1_score']
            auc = top1_garbage.loc[model, 'auc_roc']
            print(f"  {model:20}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            
        print("\nTOP1 Performance Differences (Original - Garbage):")
        for model in top1_original.index:
            acc_diff = top1_original.loc[model, 'accuracy'] - top1_garbage.loc[model, 'accuracy']
            f1_diff = top1_original.loc[model, 'f1_score'] - top1_garbage.loc[model, 'f1_score']
            auc_diff = top1_original.loc[model, 'auc_roc'] - top1_garbage.loc[model, 'auc_roc']
            print(f"  {model:20}: Δ Accuracy={acc_diff:+.3f}, Δ F1={f1_diff:+.3f}, Δ AUC={auc_diff:+.3f}")
            
    except Exception as e:
        print(f"Error loading TOP1 results: {e}")
    
    print()
    print("=" * 80)
    print("INTERPRETATION OF RESULTS")
    print("=" * 80)
    print()
    print("🔍 What this test reveals:")
    print("  • Positive differences (Original > Garbage): Models learn meaningful patterns")
    print("  • Near-zero differences: Models may be overfitting or features not informative")
    print("  • Negative differences (Garbage > Original): Unexpected, possible data leakage")
    print()
    print("📊 Key Insights:")
    print("  • Larger performance drops indicate stronger feature learning")
    print("  • Consistent drops across models suggest robust pattern detection")
    print("  • Model-specific drops reveal which algorithms work best with each feature type")
    print()
    
    print("🎯 CONCLUSION:")
    print("=" * 80)
    
    # TOP5 Analysis
    try:
        top5_avg_auc_diff = (top5_original['auc_roc'] - top5_garbage['auc_roc']).mean()
        top5_avg_f1_diff = (top5_original['f1_score'] - top5_garbage['f1_score']).mean()
        
        print(f"TOP5 Cluster Features:")
        print(f"  Average AUC drop: {top5_avg_auc_diff:.3f}")
        print(f"  Average F1 drop: {top5_avg_f1_diff:.3f}")
        
        if top5_avg_auc_diff > 0.05:
            print("  ✅ STRONG evidence that models learn meaningful cluster patterns")
        elif top5_avg_auc_diff > 0.02:
            print("  ⚠️  MODERATE evidence of pattern learning")
        else:
            print("  ❌ WEAK evidence - cluster features may not be very informative")
    except:
        print("TOP5 analysis failed")
    
    # TOP1 Analysis
    try:
        top1_avg_auc_diff = (top1_original['auc_roc'] - top1_garbage['auc_roc']).mean()
        top1_avg_f1_diff = (top1_original['f1_score'] - top1_garbage['f1_score']).mean()
        
        print(f"\nTOP1 Action Class Features:")
        print(f"  Average AUC drop: {top1_avg_auc_diff:.3f}")
        print(f"  Average F1 drop: {top1_avg_f1_diff:.3f}")
        
        if top1_avg_auc_diff > 0.05:
            print("  ✅ STRONG evidence that models learn meaningful action patterns")
        elif top1_avg_auc_diff > 0.02:
            print("  ⚠️  MODERATE evidence of pattern learning")
        else:
            print("  ❌ WEAK evidence - action features may not be very informative")
    except:
        print("TOP1 analysis failed")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    load_and_compare_results()