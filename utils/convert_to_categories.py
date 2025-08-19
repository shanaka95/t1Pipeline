#!/usr/bin/env python3
"""
Convert action class counts to category counts.
Maps the 52 action classes to their categories and aggregates the counts.
"""

import pandas as pd
import numpy as np

def load_action_category_mapping(categories_file):
    """
    Load the mapping from action classes to categories
    """
    print("📖 Loading action category mapping...")
    categories_df = pd.read_csv(categories_file)
    
    # Create mapping dictionary: action_class_id -> category
    action_to_category = {}
    for _, row in categories_df.iterrows():
        action_id = row['ID']
        category = row['Category']
        action_to_category[action_id] = category
    
    print(f"✅ Loaded mapping for {len(action_to_category)} action classes")
    print(f"📋 Categories found: {sorted(set(action_to_category.values()))}")
    
    return action_to_category

def convert_action_counts_to_categories(action_counts_file, action_to_category, output_file):
    """
    Convert action class counts to category counts
    """
    print("🔄 Loading action class counts...")
    action_df = pd.read_csv(action_counts_file)
    
    # Get action class columns
    action_columns = [col for col in action_df.columns if col.startswith('action_class_')]
    print(f"📊 Processing {len(action_columns)} action classes for {len(action_df)} videos")
    
    # Create category columns
    categories = sorted(set(action_to_category.values()))
    category_columns = [f'category_{cat}' for cat in categories]
    
    # Initialize category counts
    for cat in categories:
        action_df[f'category_{cat}'] = 0
    
    # Map action classes to categories and sum counts
    print("🔄 Converting action classes to categories...")
    for action_col in action_columns:
        # Extract action class ID from column name
        action_id = int(action_col.replace('action_class_', ''))
        
        if action_id in action_to_category:
            category = action_to_category[action_id]
            category_col = f'category_{category}'
            
            # Add the action class counts to the corresponding category
            action_df[category_col] += action_df[action_col]
        else:
            print(f"⚠️  Warning: Action class {action_id} not found in mapping")
    
    # Create new dataframe with only video_name and category columns
    result_columns = ['video_name'] + category_columns
    category_df = action_df[result_columns].copy()
    
    # Calculate total segments per video for verification
    category_df['total_segments'] = category_df[category_columns].sum(axis=1)
    
    # Save to CSV
    category_df.to_csv(output_file, index=False)
    print(f"💾 Category counts saved to: {output_file}")
    
    return category_df

def analyze_category_distribution(category_df):
    """
    Analyze the distribution of categories
    """
    print("\n📈 Category Distribution Analysis:")
    
    # Get category columns (excluding video_name and total_segments)
    category_columns = [col for col in category_df.columns if col.startswith('category_')]
    
    # Calculate total counts for each category
    category_totals = category_df[category_columns].sum().sort_values(ascending=False)
    
    print(f"🏆 Category Distribution (Total across all videos):")
    for category_col, total in category_totals.items():
        category = category_col.replace('category_', '')
        percentage = (total / category_totals.sum()) * 100
        print(f"   • Category {category}: {total:6,} segments ({percentage:5.1f}%)")
    
    # Show videos with most diverse categories
    print(f"\n🎭 Videos with Most Category Diversity:")
    category_df['num_categories'] = (category_df[category_columns] > 0).sum(axis=1)
    diverse_videos = category_df.nlargest(5, 'num_categories')[['video_name', 'num_categories', 'total_segments']]
    for _, row in diverse_videos.iterrows():
        print(f"   • {row['video_name']}: {row['num_categories']} categories, {row['total_segments']} total segments")
    
    # Show category-specific statistics
    print(f"\n📊 Category-Specific Statistics:")
    for category_col in category_columns:
        category = category_col.replace('category_', '')
        category_data = category_df[category_col]
        
        videos_with_category = (category_data > 0).sum()
        max_segments = category_data.max()
        avg_segments = category_data.mean()
        
        print(f"   • Category {category}:")
        print(f"     - Videos with this category: {videos_with_category}/{len(category_df)} ({videos_with_category/len(category_df)*100:.1f}%)")
        print(f"     - Max segments per video: {max_segments}")
        print(f"     - Average segments per video: {avg_segments:.1f}")
    
    return category_totals

def verify_category_counts(category_df, expected_total=165050):
    """
    Verify that category counts sum up correctly
    """
    print(f"\n🔍 Verifying category counts...")
    
    # Get category columns
    category_columns = [col for col in category_df.columns if col.startswith('category_')]
    
    # Calculate total segments
    total_segments = category_df['total_segments'].sum()
    category_sum = category_df[category_columns].sum().sum()
    
    print(f"📊 Verification Results:")
    print(f"   • Total segments (from total_segments column): {total_segments:,}")
    print(f"   • Total segments (sum of all categories): {category_sum:,}")
    print(f"   • Expected total: {expected_total:,}")
    print(f"   • All match: {'✅ YES' if total_segments == category_sum == expected_total else '❌ NO'}")
    
    return total_segments == expected_total

def main():
    # File paths
    categories_file = 'datasets/actions_with_categories.csv'
    action_counts_file = 'datasets/video_action_class_counts.csv'
    output_file = 'datasets/video_category_counts.csv'
    
    print("🚀 Starting action class to category conversion...")
    
    # Step 1: Load action category mapping
    action_to_category = load_action_category_mapping(categories_file)
    
    # Step 2: Convert action counts to category counts
    category_df = convert_action_counts_to_categories(
        action_counts_file, 
        action_to_category, 
        output_file
    )
    
    # Step 3: Analyze category distribution
    category_totals = analyze_category_distribution(category_df)
    
    # Step 4: Verify counts
    verification_passed = verify_category_counts(category_df)
    
    print(f"\n✅ Conversion completed!")
    if verification_passed:
        print(f"🎉 SUCCESS: Category conversion verified! Total segments: {category_df['total_segments'].sum():,}")
    else:
        print(f"⚠️  WARNING: Verification failed. Please check the data.")
    
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    main()

