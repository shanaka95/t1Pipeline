#!/usr/bin/env python3
"""
Verify action class counts in the CSV file.
Check if counts sum up to total segments per video and overall total.
"""

import pandas as pd
import numpy as np

def verify_action_class_counts(csv_file):
    """
    Verify that action class counts sum up correctly
    """
    print("🔍 Loading CSV file...")
    df = pd.read_csv(csv_file)
    
    print(f"📊 CSV file loaded: {len(df)} videos, {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Calculate total segments per video (sum of all action classes)
    action_columns = [col for col in df.columns if col.startswith('action_class_')]
    print(f"\n🎯 Action class columns: {len(action_columns)} (0-51)")
    
    # Calculate sum for each video
    df['calculated_total'] = df[action_columns].sum(axis=1)
    
    # Check for any videos with zero total segments
    zero_total_videos = df[df['calculated_total'] == 0]
    if len(zero_total_videos) > 0:
        print(f"\n⚠️  WARNING: Found {len(zero_total_videos)} videos with zero total segments:")
        for _, row in zero_total_videos.iterrows():
            print(f"   • {row['video_name']}")
    
    # Calculate overall total
    total_segments = df['calculated_total'].sum()
    expected_total = 165050
    
    print(f"\n📈 Verification Results:")
    print(f"   • Total segments calculated: {total_segments:,}")
    print(f"   • Expected total: {expected_total:,}")
    print(f"   • Difference: {total_segments - expected_total:,}")
    print(f"   • Match: {'✅ YES' if total_segments == expected_total else '❌ NO'}")
    
    # Show videos with most and least segments
    print(f"\n📹 Videos with Most Segments:")
    top_videos = df.nlargest(5, 'calculated_total')[['video_name', 'calculated_total']]
    for _, row in top_videos.iterrows():
        print(f"   • {row['video_name']}: {row['calculated_total']:,} segments")
    
    print(f"\n📹 Videos with Least Segments (non-zero):")
    non_zero_videos = df[df['calculated_total'] > 0]
    bottom_videos = non_zero_videos.nsmallest(5, 'calculated_total')[['video_name', 'calculated_total']]
    for _, row in bottom_videos.iterrows():
        print(f"   • {row['video_name']}: {row['calculated_total']:,} segments")
    
    # Check action class distribution
    print(f"\n🏆 Action Class Distribution (Total across all videos):")
    action_totals = df[action_columns].sum().sort_values(ascending=False)
    print(f"   • Most frequent: {action_totals.index[0]} = {action_totals.iloc[0]:,} segments")
    print(f"   • Least frequent (non-zero): {action_totals[action_totals > 0].index[-1]} = {action_totals[action_totals > 0].iloc[-1]:,} segments")
    print(f"   • Zero frequency classes: {sum(action_totals == 0)}")
    
    # Show top 10 action classes
    print(f"\n🏆 Top 10 Most Frequent Action Classes:")
    for i, (action_class, count) in enumerate(action_totals.head(10).items()):
        percentage = (count / total_segments) * 100
        print(f"   {i+1:2d}. {action_class}: {count:6,} segments ({percentage:5.1f}%)")
    
    # Check for any negative values (shouldn't exist)
    negative_values = df[action_columns].lt(0).any().any()
    if negative_values:
        print(f"\n❌ ERROR: Found negative values in action class counts!")
    else:
        print(f"\n✅ No negative values found in action class counts")
    
    # Save verification results
    verification_file = csv_file.replace('.csv', '_verification.csv')
    df.to_csv(verification_file, index=False)
    print(f"\n💾 Verification results saved to: {verification_file}")
    
    return {
        'total_segments': total_segments,
        'expected_total': expected_total,
        'match': total_segments == expected_total,
        'num_videos': len(df),
        'num_zero_videos': len(zero_total_videos),
        'action_class_totals': action_totals
    }

def main():
    csv_file = 'datasets/video_action_class_counts.csv'
    
    print("🚀 Starting action class count verification...")
    results = verify_action_class_counts(csv_file)
    
    print(f"\n✅ Verification completed!")
    if results['match']:
        print(f"🎉 SUCCESS: All counts are correct! Total segments: {results['total_segments']:,}")
    else:
        print(f"⚠️  WARNING: Count mismatch detected. Please check the data.")

if __name__ == "__main__":
    main()

