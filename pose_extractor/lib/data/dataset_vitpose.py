import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from pose_extractor.lib.utils.utils_data import crop_scale

def read_input(npz_path, vid_size, scale_range, focus):
    # Load the keypoints from the npz file
    kpts_all = np.load(npz_path)['keypoints']
    scale_range =[1,1]

    # Check and fix dimensions if necessary
    if len(kpts_all.shape) == 4:  # If shape is [person, frames, joints, coords]
        # Take the first person only if there are multiple
        kpts_all = kpts_all[0]
    
    # Ensure we have [frames, joints, coords] format
    assert len(kpts_all.shape) == 3, f"Expected keypoints shape [frames, joints, coords], got {kpts_all.shape}"
    
    w, h = vid_size
    scale = min(w,h) / 2.0
    
    # Normalize keypoints
    # We only modify the x,y coordinates (first 2 elements), not the confidence
    kpts_all[:,:,0] = kpts_all[:,:,0] - w / 2.0
    kpts_all[:,:,1] = kpts_all[:,:,1] - h / 2.0
    kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
    
    motion = kpts_all

    if scale_range:
        motion = crop_scale(kpts_all, scale_range) 
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, npz_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.npz_path = npz_path
        self.clip_len = clip_len
        
        # Ensure vid_size is provided
        if vid_size is None:
            # Try to get vid_size from the npz file
            try:
                data = np.load(npz_path)
                if 'vid_size' in data:
                    vid_size = data['vid_size']
                    print(f"Using video size from npz: {vid_size}")
            except:
                raise ValueError("Video size must be provided either directly or in the npz file")
                
        self.vid_all = read_input(npz_path, vid_size, scale_range, focus)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]