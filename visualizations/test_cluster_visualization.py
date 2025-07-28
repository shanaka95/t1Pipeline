#!/usr/bin/env python3
"""
Test script for cluster visualization with poses directory.
Allows specifying base directory and cluster number.
"""

import os
import sys
import argparse
from visualize_cluster_with_poses_dir import visualize_clusters_with_poses_dir

def test_visualization_with_cluster(
    poses_dir="/home/janus/iwso-datasets/t1-body-poses-final/",
    clustering_base_dir="/home/shanaka/Desktop/thesis/pipeline-final/test/clustering_info_with_top5_labels",
    cluster_number=99,
    output_dir="test_cluster_visualizations",
    num_segments=2,
    fps=10
):
    """
    Test the cluster visualization with specific cluster number.
    
    Parameters:
    poses_dir (str): Master directory containing all pose directories
    clustering_base_dir (str): Base directory containing clustering results
    cluster_number (int): Specific cluster number to use
    output_dir (str): Output directory for visualizations
    num_segments (int): Number of segments per cluster for testing
    fps (int): Frames per second for animations
    """
    
    print("🧪 Testing cluster visualization...")
    print(f"📂 Poses directory: {poses_dir}")
    print(f"📊 Clustering base directory: {clustering_base_dir}")
    print(f"🎯 Using cluster number: {cluster_number}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎬 Segments per cluster: {num_segments}")
    
    # Check if clustering directory exists
    clustering_dir = os.path.join(clustering_base_dir, str(cluster_number))
    if not os.path.exists(clustering_dir):
        print(f"❌ Clustering directory not found: {clustering_dir}")
        print(f"Available directories in {clustering_base_dir}:")
        try:
            available = [d for d in os.listdir(clustering_base_dir) if os.path.isdir(os.path.join(clustering_base_dir, d))]
            for d in sorted(available):
                print(f"   - {d}")
        except Exception as e:
            print(f"   Error listing directories: {e}")
        return False
    
    # Check if clustering summary exists
    clustering_summary = os.path.join(clustering_dir, "clustering_summary.json")
    if not os.path.exists(clustering_summary):
        print(f"❌ Clustering summary not found: {clustering_summary}")
        return False
    
    print(f"✅ Found clustering data for {cluster_number} clusters")
    
    # Test with specified parameters
    try:
        output_dir = visualize_clusters_with_poses_dir(
            poses_master_dir=poses_dir,
            clustering_base_dir=clustering_base_dir,
            cluster_number=cluster_number,  # Specify exact cluster number
            output_dir=output_dir,
            num_segments_per_cluster=num_segments,
            fps=fps
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Test results saved in: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Test cluster visualization with specific parameters")
    
    parser.add_argument("--poses_dir", 
                       default="/home/janus/iwso-datasets/t1-body-poses-final/",
                       help="Master directory containing all pose directories")
    
    parser.add_argument("--clustering_base_dir", 
                       default="/home/shanaka/Desktop/thesis/pipeline-final/test/clustering_info_with_top5_labels",
                       help="Base directory containing clustering results")
    
    parser.add_argument("--cluster_number", 
                       type=int, 
                       default=99,
                       help="Specific cluster number to use for visualization")
    
    parser.add_argument("--output_dir", 
                       default="test_cluster_visualizations",
                       help="Output directory for visualizations")
    
    parser.add_argument("--num_segments", 
                       type=int, 
                       default=2,
                       help="Number of segments per cluster for testing")
    
    parser.add_argument("--fps", 
                       type=int, 
                       default=10,
                       help="Frames per second for animations")
    
    parser.add_argument("--list_clusters", 
                       action="store_true",
                       help="List available cluster numbers and exit")
    
    args = parser.parse_args()
    
    # List available clusters if requested
    if args.list_clusters:
        print(f"Available cluster numbers in {args.clustering_base_dir}:")
        try:
            available = [d for d in os.listdir(args.clustering_base_dir) 
                        if os.path.isdir(os.path.join(args.clustering_base_dir, d))]
            for d in sorted(available, key=lambda x: int(x) if x.isdigit() else 0):
                cluster_dir = os.path.join(args.clustering_base_dir, d)
                summary_file = os.path.join(cluster_dir, "clustering_summary.json")
                if os.path.exists(summary_file):
                    print(f"   ✅ {d} (has clustering_summary.json)")
                else:
                    print(f"   ⚠️  {d} (missing clustering_summary.json)")
        except Exception as e:
            print(f"   Error listing directories: {e}")
        return 0
    
    # Run the test
    success = test_visualization_with_cluster(
        poses_dir=args.poses_dir,
        clustering_base_dir=args.clustering_base_dir,
        cluster_number=args.cluster_number,
        output_dir=args.output_dir,
        num_segments=args.num_segments,
        fps=args.fps
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 