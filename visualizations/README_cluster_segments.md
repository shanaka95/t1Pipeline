# Cluster Segment Visualization Script

## Overview

`visualize_cluster_segments.py` is a script that creates 3D pose visualizations for random segments from specified clusters. It reads clustering data, identifies segments from a chosen cluster, loads the corresponding 3D pose data, and creates dual-view GIF animations.

## Features

- **Smart Video Prioritization**: Automatically prioritizes segments from videos with available pose data
- **Dual-View Animations**: Creates side-by-side 2D and 3D visualizations
- **Flexible Configuration**: Configurable number of segments, FPS, output directory, and random seed
- **Robust Error Handling**: Gracefully handles missing pose data and provides clear feedback
- **Progress Tracking**: Detailed progress output with emoji indicators

## Data Structure Requirements

### Clustering Data File
The script expects a pickle file containing clustering results with this structure:
```python
{
    'cluster_summary': {
        0: {
            'total_segments': int,
            'num_videos': int, 
            'videos': {
                'video_name': [segment_id_1, segment_id_2, ...],
                ...
            },
            'cluster_center': [float, ...]
        },
        1: { ... },
        ...
    }
}
```

### Poses Directory Structure
```
poses/
├── video_name_1/
│   └── poses_3D.pkl
├── video_name_2/
│   └── poses_3D.pkl
└── ...
```

Each `poses_3D.pkl` file should contain a list of numpy arrays with shape `(1, 243, 17, 3)` representing:
- 1: batch dimension (removed during processing)
- 243: number of frames
- 17: number of joints (H36M skeleton)
- 3: x, y, z coordinates

## Usage

### Basic Usage
```bash
python visualize_cluster_segments.py \
    --clustering_data kmeans_top5_labels_results.pkl \
    --poses_dir poses \
    --cluster_id 15
```

### Advanced Usage
```bash
python visualize_cluster_segments.py \
    --clustering_data results.pkl \
    --poses_dir /path/to/poses \
    --cluster_id 42 \
    --num_segments 3 \
    --fps 20 \
    --output_dir visualizations \
    --seed 123
```

### Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--clustering_data` | `-c` | Yes | - | Path to clustering data pickle file |
| `--poses_dir` | `-p` | Yes | - | Root directory containing pose data |
| `--cluster_id` | `-i` | Yes | - | ID of the cluster to visualize (0-99) |
| `--output_dir` | `-o` | No | `cluster_visualizations` | Output directory for visualizations |
| `--num_segments` | `-n` | No | `5` | Number of segments to visualize |
| `--fps` | `-f` | No | `15` | Frames per second for animations |
| `--seed` | `-s` | No | None | Random seed for reproducible results |

## Output

The script creates:
- A directory structure: `{output_dir}/cluster_{cluster_id:03d}/`
- GIF files named: `cluster_{cluster_id:03d}_seg_{i:02d}_{video_name}_segment_{segment_id}.gif`
- Each GIF contains dual-view animations (2D side view + 3D isometric view)

## Examples

### Example 1: Basic cluster visualization
```bash
python visualize_cluster_segments.py -c data.pkl -p poses -i 0
```

### Example 2: Create 3 visualizations with custom settings
```bash
python visualize_cluster_segments.py \
    -c kmeans_results.pkl \
    -p /data/poses \
    -i 25 \
    -n 3 \
    -f 20 \
    -o my_visualizations \
    -s 42
```

## Key Features

### Smart Video Selection
The script automatically:
- Checks which videos in the cluster have available pose data
- Prioritizes segments from videos with available poses
- Falls back to random selection if no poses are available
- Provides clear indicators (📁 for available, ❓ for unavailable)

### Robust Error Handling
- Continues processing even if some segments fail
- Provides detailed error messages and warnings
- Gracefully handles missing pose files
- Reports success/failure statistics at the end

### Detailed Progress Output
- 🚀 Starting visualization
- 📊 Loading clustering data
- 🎯 Selecting segments
- 📹 Processing each segment
- 🎬 Creating animations
- ✅ Success confirmations

## Technical Details

### H36M Skeleton Structure
The script uses the Human3.6M skeleton with 17 joints:
0. Hip, 1. Right Hip, 2. Right Knee, 3. Right Ankle
4. Left Hip, 5. Left Knee, 6. Left Ankle
7. Spine, 8. Thorax, 9. Neck, 10. Head
11. Left Shoulder, 12. Left Elbow, 13. Left Hand
14. Right Shoulder, 15. Right Elbow, 16. Right Hand

### Animation Features
- **2D Side View**: X-axis (left-right), Y-axis (height, flipped)
- **3D Isometric View**: Full 3D visualization with 15° elevation, 45° azimuth
- **Frame Counter**: Shows current frame and progress percentage
- **Joint Highlighting**: Red dots mark joint positions in 3D view

## Dependencies

```python
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import random
from typing import List, Tuple, Dict, Any
```

## Error Scenarios

The script handles these common issues:
- Missing clustering data file
- Missing poses directory
- Invalid cluster ID
- Missing pose files for specific videos
- Corrupted pose data
- Invalid segment IDs

## Performance Notes

- GIF file sizes are typically 4-5MB per visualization
- Processing time depends on number of frames (default 243) and FPS
- Memory usage scales with number of segments processed simultaneously
- Animation creation is the most time-intensive step

## Testing

To test the script with available data:
```bash
# Test with a small cluster
python visualize_cluster_segments.py -c data.pkl -p poses -i 0 -n 1

# Test with reproducible results
python visualize_cluster_segments.py -c data.pkl -p poses -i 0 -s 42
```

The script will automatically prioritize segments from videos with available pose data, making it easy to test locally even with partial datasets. 