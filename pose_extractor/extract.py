import os
import numpy as np
import sys
from tqdm import tqdm
import imageio
import torch
from pose_extractor import vitpose
import torch.nn as nn
from torch.utils.data import DataLoader
from pose_extractor.lib.utils.tools import *
from pose_extractor.lib.utils.learning import *
from pose_extractor.lib.utils.utils_data import flip_data
from pose_extractor.lib.data.dataset_vitpose import WildDetDataset
from pose_extractor.lib.utils.vismo import render_and_save
 
def extract_pose(vid_path, out_path):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load bounding boxes
    bbox_file = "preprocess_videos/bounding_boxes.npy"
    bounding_boxes = np.load(bbox_file, allow_pickle=True).item()
    
    # Get video filename without extension
    video_filename = os.path.basename(vid_path)
    video_name = os.path.splitext(video_filename)[0]
    
    # Select bounding box for current video
    if video_name in bounding_boxes:
        bbox_info = bounding_boxes[video_name]
        bounding_box = bbox_info['bounding_box']
        print(f"Found bounding box for video '{video_name}': {bounding_box}")
        print(f"Video dimensions: {bbox_info['width']}x{bbox_info['height']}")
    else:
        print(f"Warning: No bounding box found for video '{video_name}'")
        print(f"Available videos: {list(bounding_boxes.keys())}")
        bounding_box = None

    # Load MotionBERT config
    config = "pose_extractor/configs/pose3d/MB_ft_h36m.yaml"

    # Load MotionBERT checkpoint
    model_path = "pose_extractor/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"

    # Load keypoints
    npz_path = out_path + "/keypoints_2d.npz"

    # Create output directory
    os.makedirs(out_path, exist_ok=True)

    print(f"Generating 2D keypoints for {vid_path}")

    all_frame_poses, frame_count, person_detected_frame, vid_size = vitpose.generate_2d_pose(vid_path, bounding_box)
        
    # Save the keypoints to a npz file
    np.savez(npz_path, 
           keypoints=all_frame_poses, 
           frame_count=frame_count, 
           person_detected_frame=person_detected_frame,
           vid_size=vid_size)
        
    print(f"Processed {frame_count} frames")
    print(f"Person detected from frame {person_detected_frame}")
    print(f"Saved keypoints with shape {all_frame_poses.shape} in H36M format")

    # Get config
    args = get_config(config)

    # Load MotionBERT model
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print(f"Generating 3D poses")
    # Load checkpoint
    print('Loading checkpoint', model_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()

    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }

    # Load video metadata
    try:
        vid = imageio.get_reader(vid_path, 'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        print(f"Video FPS: {fps_in}")
    except Exception as e:
        print(f"Warning: Could not load video metadata: {e}")
        fps_in = 30
        print(f"Using default FPS: {fps_in}")

    # Ensure output directory exists
    os.makedirs(out_path, exist_ok=True)

    # Load the keypoints data
    try:
        keypoints_data = np.load(npz_path)
        print(f"Loaded keypoints with keys: {keypoints_data.files}")
        
        if 'keypoints' in keypoints_data:
            keypoint_shape = keypoints_data['keypoints'].shape
            print(f"Keypoint shape: {keypoint_shape}")
        
        if 'vid_size' in keypoints_data:
            vid_size = keypoints_data['vid_size']
            print(f"Video size from npz: {vid_size}")
        else:
            print("Warning: No video size in keypoints file")
            vid_size = None
    except Exception as e:
        print(f"Error loading keypoints: {e}")
        raise

    # Create dataset
    try:
        print(f"Creating dataset with video size: {vid_size}")
        wild_dataset = WildDetDataset(npz_path, vid_size=vid_size)
        print(f"Dataset created with {len(wild_dataset)} clips")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise

    test_loader = DataLoader(wild_dataset, **testloader_params)

    # Process data through MotionBERT
    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            print(f"Processing batch with shape: {batch_input.shape}")
            
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    # Process results
    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    print(f"Final 3D pose shape: {results_all.shape}")

    #print(f"Rendering and saving video")

    filename = os.path.basename(vid_path)
    filename = filename.split('.')[0]

    # Render and save video
    #render_and_save(results_all, '%s/%s_X3D.mp4' % (out_path, filename), keep_imgs=False, fps=fps_in)

    # Save 3D pose data
    np.save('%s/%s_X3D.npy' % (out_path, filename), results_all)
    print(f"3D pose data saved to {out_path}/{filename}_X3D.npy")
    #print(f"3D pose video saved to {out_path}/{filename}_X3D.mp4")