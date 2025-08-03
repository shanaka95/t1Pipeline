"""
Comprehensive Comparison: Top1 Action Class vs Top5 Clustering Approaches
This script compares the performance of depression prediction models using:
1. Top1 Action Class approach (direct action labeling)
2. Top5 Clustering approach (k-means clustering)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

def load_top5_results():
    """Load the most recent top5 clustering results"""
    print("Loading Top5 Clustering Results...")
    
    # Find the most recent results
    saved_models_dir = 'saved_models'
    evaluation_files = [f for f in os.listdir(saved_models_dir) if f.startswith('comprehensive_evaluation_')]
    
    if not evaluation_files:
        print("No top5 clustering results found")
        return None, None
        
    # Get the most recent file
    latest_eval_file = sorted(evaluation_files)[-1]
    timestamp = latest_eval_file.replace('comprehensive_evaluation_', '').replace('.csv', '')
    
    # Load evaluation results
    eval_path = os.path.join(saved_models_dir, latest_eval_file)
    top5_results = pd.read_csv(eval_path, index_col=0)
    
    # Load workflow summary
    summary_path = os.path.join(saved_models_dir, f'workflow_summary_{timestamp}.json')
    with open(summary_path, 'r') as f:
        top5_summary = json.load(f)
    
    print(f"Loaded Top5 results from timestamp: {timestamp}")
    print(f"Models evaluated: {top5_summary['models_trained']}")
    
    return top5_results, top5_summary

def load_top1_results():
    """Load the most recent top1 action class results"""
    print("Loading Top1 Action Class Results...")
    
    # Find the most recent top1 results directory
    result_dirs = [d for d in os.listdir('.') if d.startswith('top1_prediction_results_')]
    
    if not result_dirs:
        print("No top1 results found")
        return None, None
        
    latest_dir = sorted(result_dirs)[-1]
    
    # Load workflow summary
    summary_path = os.path.join(latest_dir, 'top1_workflow_summary.json')
    with open(summary_path, 'r') as f:
        top1_summary = json.load(f)
    
    # Extract binary classification results
    binary_results = {}
    for config in ['binary_smote', 'binary_none']:
        if config in top1_summary['workflow_results']:
            config_results = top1_summary['workflow_results'][config]['model_metrics']
            for model, metrics in config_results.items():
                key = f"{model}_{config}"
                binary_results[key] = metrics
    
    print(f"Loaded Top1 results from: {latest_dir}")
    print(f"Configurations: {list(top1_summary['workflow_results'].keys())}")
    
    return binary_results, top1_summary

def create_comparison_dataframe(top5_results, top1_results):
    """Create a comprehensive comparison dataframe"""
    print("Creating Comparison DataFrame...")
    
    comparison_data = []
    
    # Add Top5 Clustering results
    if top5_results is not None:
        for model in top5_results.index:
            comparison_data.append({
                'Approach': 'Top5 Clustering',
                'Model': model,
                'Configuration': 'default',
                'Accuracy': top5_results.loc[model, 'accuracy'],
                'Precision': top5_results.loc[model, 'precision'],
                'Recall': top5_results.loc[model, 'recall'],
                'F1-Score': top5_results.loc[model, 'f1_score'],
                'ROC-AUC': top5_results.loc[model, 'auc_roc'],
                'PR-AUC': top5_results.loc[model, 'avg_precision']
            })
    
    # Add Top1 Action Class results
    if top1_results is not None:
        for model_config, metrics in top1_results.items():
            model_name = model_config.split('_')[0] + ' ' + model_config.split('_')[1]
            config = '_'.join(model_config.split('_')[2:])
            
            comparison_data.append({
                'Approach': 'Top1 Action Class',
                'Model': model_name,
                'Configuration': config,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', np.nan),
                'PR-AUC': metrics.get('pr_auc', np.nan)
            })
    
    df = pd.DataFrame(comparison_data)
    return df

def create_comparison_visualizations(df):
    """Create comprehensive comparison visualizations"""
    print("Creating Comparison Visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. Overall Performance Comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Top1 Action Class vs Top5 Clustering - Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        
        # Create grouped bar plot
        df_metric = df.pivot_table(values=metric, index='Model', columns='Approach', aggfunc='mean')
        df_metric.plot(kind='bar', ax=ax, alpha=0.8, width=0.8)
        
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        ax.legend(title='Approach', loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('top1_vs_top5_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Best Model Comparison by Configuration
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best models for each approach
    best_models = df.groupby(['Approach', 'Configuration'])['F1-Score'].max().reset_index()
    best_models_detailed = df.merge(best_models, on=['Approach', 'Configuration', 'F1-Score'])
    
    # Plot F1-Score comparison
    sns.barplot(data=best_models_detailed, x='Configuration', y='F1-Score', 
                hue='Approach', ax=axes[0])
    axes[0].set_title('Best F1-Score by Configuration', fontweight='bold')
    axes[0].set_ylabel('F1-Score')
    axes[0].legend(title='Approach')
    
    # Plot ROC-AUC comparison
    sns.barplot(data=best_models_detailed.dropna(subset=['ROC-AUC']), 
                x='Configuration', y='ROC-AUC', hue='Approach', ax=axes[1])
    axes[1].set_title('Best ROC-AUC by Configuration', fontweight='bold')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend(title='Approach')
    
    plt.tight_layout()
    plt.savefig('top1_vs_top5_best_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed Metrics Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create a pivot table for heatmap
    df_pivot = df.pivot_table(values=['Accuracy', 'F1-Score', 'ROC-AUC'], 
                             index=['Approach', 'Model', 'Configuration'], 
                             aggfunc='mean')
    
    sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Detailed Performance Metrics Heatmap', fontweight='bold')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Model Configuration')
    
    plt.tight_layout()
    plt.savefig('top1_vs_top5_detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_differences(top1_summary):
    """Analyze the differences in feature types between approaches"""
    print("Analyzing Feature Differences...")
    
    analysis = {
        'Top5 Clustering Approach': {
            'feature_type': 'Cluster percentages',
            'num_features': 100,  # Typical number from previous results
            'interpretability': 'Abstract clusters',
            'feature_meaning': 'Percentage of video time in each cluster',
            'clustering_method': 'K-means on top 5 action labels'
        },
        'Top1 Action Class Approach': {
            'feature_type': 'Action class percentages',
            'num_features': top1_summary['dataset_info']['action_features'],
            'interpretability': 'Concrete human actions',
            'feature_meaning': 'Percentage of video time in each action class',
            'clustering_method': 'Direct assignment to action classes'
        }
    }
    
    # Top action classes analysis
    top_actions = top1_summary['top_action_classes']
    action_correlations = {k: v for k, v in top1_summary['action_depression_correlations'].items() 
                          if not pd.isna(v)}
    
    analysis['Top1 Action Class Approach']['top_features'] = list(top_actions.keys())[:5]
    analysis['Top1 Action Class Approach']['most_predictive'] = list(action_correlations.keys())[:5]
    
    return analysis

def create_summary_report(df, top1_summary, top5_summary, feature_analysis):
    """Create a comprehensive summary report"""
    print("Creating Summary Report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'top1_vs_top5_comparison_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE COMPARISON REPORT\n")
        f.write("Top1 Action Class vs Top5 Clustering Approaches\n")
        f.write("="*80 + "\n\n")
        
        # Dataset information
        f.write("1. DATASET INFORMATION\n")
        f.write("-"*30 + "\n")
        if top1_summary:
            f.write(f"Top1 Dataset:\n")
            f.write(f"  - Videos: {top1_summary['dataset_info']['shape'][0]}\n")
            f.write(f"  - Patients: {top1_summary['dataset_info']['patients']}\n")
            f.write(f"  - Features: {top1_summary['dataset_info']['action_features']}\n")
            f.write(f"  - Depression distribution: {top1_summary['target_distributions']['binary']}\n\n")
        
        # Approach comparison
        f.write("2. APPROACH COMPARISON\n")
        f.write("-"*30 + "\n")
        for approach, details in feature_analysis.items():
            f.write(f"{approach}:\n")
            for key, value in details.items():
                if key != 'top_features' and key != 'most_predictive':
                    f.write(f"  - {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
        
        # Performance summary
        f.write("3. PERFORMANCE SUMMARY\n")
        f.write("-"*30 + "\n")
        
        # Best performance by approach
        best_by_approach = df.groupby('Approach').agg({
            'Accuracy': 'max',
            'F1-Score': 'max',
            'ROC-AUC': 'max'
        })
        
        f.write("Best Performance by Approach:\n")
        for approach in best_by_approach.index:
            f.write(f"\n{approach}:\n")
            f.write(f"  - Best Accuracy: {best_by_approach.loc[approach, 'Accuracy']:.4f}\n")
            f.write(f"  - Best F1-Score: {best_by_approach.loc[approach, 'F1-Score']:.4f}\n")
            f.write(f"  - Best ROC-AUC: {best_by_approach.loc[approach, 'ROC-AUC']:.4f}\n")
        
        # Model-wise comparison
        f.write("\n4. MODEL-WISE COMPARISON\n")
        f.write("-"*30 + "\n")
        model_comparison = df.groupby(['Model', 'Approach']).agg({
            'F1-Score': 'mean',
            'ROC-AUC': 'mean'
        }).round(4)
        
        f.write(model_comparison.to_string())
        f.write("\n\n")
        
        # Key insights
        f.write("5. KEY INSIGHTS\n")
        f.write("-"*30 + "\n")
        
        # Find best overall model
        best_overall = df.loc[df['F1-Score'].idxmax()]
        f.write(f"Best Overall Model: {best_overall['Model']} ({best_overall['Approach']})\n")
        f.write(f"  - F1-Score: {best_overall['F1-Score']:.4f}\n")
        f.write(f"  - ROC-AUC: {best_overall['ROC-AUC']:.4f}\n\n")
        
        # Compare approaches
        top1_avg = df[df['Approach'] == 'Top1 Action Class']['F1-Score'].mean()
        top5_avg = df[df['Approach'] == 'Top5 Clustering']['F1-Score'].mean()
        
        f.write(f"Average F1-Score Comparison:\n")
        f.write(f"  - Top1 Action Class: {top1_avg:.4f}\n")
        f.write(f"  - Top5 Clustering: {top5_avg:.4f}\n")
        f.write(f"  - Difference: {abs(top1_avg - top5_avg):.4f} ")
        f.write(f"({'Top1 better' if top1_avg > top5_avg else 'Top5 better'})\n\n")
        
        # Feature insights for Top1
        if top1_summary:
            f.write("6. TOP1 ACTION CLASS INSIGHTS\n")
            f.write("-"*30 + "\n")
            f.write("Most Active Action Classes:\n")
            for action, pct in list(top1_summary['top_action_classes'].items())[:5]:
                class_id = action.split('_')[-1]
                f.write(f"  - Class {class_id}: {pct:.3f} ({pct*100:.1f}%)\n")
            
            f.write("\nMost Correlated with Depression:\n")
            correlations = {k: v for k, v in top1_summary['action_depression_correlations'].items() 
                          if not pd.isna(v)}
            for action, corr in list(correlations.items())[:5]:
                class_id = action.split('_')[-1]
                f.write(f"  - Class {class_id}: {corr:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {report_file}")
    return report_file

def main():
    """Run the complete comparison analysis"""
    print("Starting Top1 vs Top5 Comparison Analysis")
    print("="*60)
    
    # Load results
    top5_results, top5_summary = load_top5_results()
    top1_results, top1_summary = load_top1_results()
    
    if top5_results is None and top1_results is None:
        print("No results found to compare!")
        return
    
    # Create comparison dataframe
    df = create_comparison_dataframe(top5_results, top1_results)
    
    if df.empty:
        print("No comparable results found!")
        return
    
    print(f"\nComparison DataFrame created with {len(df)} model configurations")
    print("\nPreview:")
    print(df.head())
    
    # Create visualizations
    create_comparison_visualizations(df)
    
    # Analyze feature differences
    feature_analysis = analyze_feature_differences(top1_summary) if top1_summary else {}
    
    # Create summary report
    report_file = create_summary_report(df, top1_summary, top5_summary, feature_analysis)
    
    # Save comparison data
    df.to_csv('top1_vs_top5_comparison_data.csv', index=False)
    
    print(f"\n{'='*60}")
    print("Comparison Analysis Completed!")
    print("Generated files:")
    print("  - top1_vs_top5_performance_comparison.png")
    print("  - top1_vs_top5_best_models_comparison.png") 
    print("  - top1_vs_top5_detailed_heatmap.png")
    print("  - top1_vs_top5_comparison_data.csv")
    print(f"  - {report_file}")
    print(f"{'='*60}")
    
    return df, report_file

if __name__ == "__main__":
    comparison_df, report_file = main()