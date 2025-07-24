#!/usr/bin/env python3
"""
Inference Script for CTR-GCN
=============================

This script performs inference on pose sequences using a trained CTR-GCN model.
It reads pose data from poses.pkl (containing 243-frame segments) and predicts
action labels using the specified configuration and checkpoint.

Usage:
    python inference.py --config config/custom/improved.yaml --weights work_dir/custom/ctrgcn_improved/runs-66-23166.pt

Author: AI Assistant
Date: 2025
"""

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F

# Import required modules
from action_recognition.ctrgcn.feeders import tools

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(__import__(mod_str, fromlist=[class_str]), class_str)

# Hardcoded config and checkpoint paths
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/custom/improved.yaml')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'work_dir/custom/ctrgcn_improved/runs-56-19656.pt')

# Device selection (CPU by default, use CUDA if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model/class cache
_model = None
_config = None
_class_names = None

def load_model():
    global _model, _config
    if _model is not None and _config is not None:
        return _model, _config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    Model = import_class(config['model'])
    model = Model(**config['model_args'])
    weights = torch.load(WEIGHTS_PATH, map_location='cpu')
    if any(key.startswith('module.') for key in weights.keys()):
        weights = {key[7:]: value for key, value in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    model = model.to(DEVICE)
    _model = model
    _config = config
    return model, config

def preprocess_pose_data(pose_data, window_size=64, p_interval=[0.95]):
    if pose_data.ndim == 4:
        data = pose_data.squeeze(0).transpose(2, 0, 1)
    elif pose_data.ndim == 3:
        data = pose_data.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unexpected pose_data shape: {pose_data.shape}")
    data = data[:, :, :, np.newaxis]
    C, T, V, M = data.shape
    data = tools.valid_crop_resize(data, T, p_interval, window_size)
    return data

def predict_label(model, data):
    data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(data_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_label].item()
    return predicted_label, confidence, probabilities.cpu().numpy()[0]

def load_class_names(config):
    global _class_names
    if _class_names is not None:
        return _class_names
    class_names = {}
    label_path = config.get('test_feeder_args', {}).get('label_path', None)
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = int(parts[1])
                    if label not in class_names:
                        class_names[label] = f"Class_{label}"
    for i in range(52):
        if i not in class_names:
            class_names[i] = f"Class_{i}"
    _class_names = class_names
    return class_names

def extract_embeddings_from_segments(segments):
    """
    Extract feature embeddings from pose segments for clustering/analysis.
    
    segments: List of np.ndarray, each of shape (243, 17, 3)
    Returns: List of dicts with keys: sequence_id, embedding (numpy array)
    """
    model, config = load_model()
    window_size = config.get('test_feeder_args', {}).get('window_size', 64)
    results = []
    
    for i, pose_data in enumerate(segments):
        processed_data = preprocess_pose_data(pose_data, window_size=window_size)
        data_tensor = torch.from_numpy(processed_data).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model.extract_embedding(data_tensor)
            embedding_np = embedding.cpu().numpy()[0]  # Remove batch dimension
        
        results.append({
            'sequence_id': i,
            'embedding': embedding_np
        })
    
    return results

def run_inference_on_segments(segments):
    """
    segments: List of np.ndarray, each of shape (243, 17, 3)
    Returns: List of dicts with keys: sequence_id, predicted_label, confidence, class_name, probabilities
    """
    model, config = load_model()
    class_names = load_class_names(config)
    window_size = config.get('test_feeder_args', {}).get('window_size', 64)
    results = []
    for i, pose_data in enumerate(segments):
        processed_data = preprocess_pose_data(pose_data, window_size=window_size)
        pred_label, confidence, prob_distribution = predict_label(model, processed_data)
        class_name = class_names.get(pred_label, f"Class_{pred_label}")
        results.append({
            'sequence_id': i,
            'predicted_label': pred_label,
            'confidence': confidence,
            'class_name': class_name,
            'probabilities': prob_distribution
        })
    return results 