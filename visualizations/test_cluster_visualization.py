#!/usr/bin/env python3
"""
Test script for cluster visualization with poses directory.
"""

import os
from visualize_cluster_with_poses_dir import visualize_clusters_with_poses_dir

def test_visualization():
    """Test the cluster visualization with a small subset."""
    
    print("🧪 Testing cluster visualization...")
    
    # Test with default parameters but smaller output
    try:
        output_dir = visualize_clusters_with_poses_dir(
            poses_master_dir="/home/janus/iwso-datasets/t1-body-poses-final/",
            clustering_base_dir="/home/shanaka/Desktop/thesis/pipeline-final/test/clustering_info_with_top5_labels",
            output_dir="test_cluster_visualizations",
            num_segments_per_cluster=2,  # Only 2 segments per cluster for testing
            fps=10  # Lower FPS for faster processing
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Test results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_visualization() 