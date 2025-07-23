import torch
import uuid, sys
import numpy as np, os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image
import cv2, json
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

def coco2h36m(x):
    '''
    Converts COCO keypoints to H36M format.
    
    COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
    
    H36M:
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    7: 'belly',
    8: 'neck',
    9: 'nose',
    10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
    '''
    # Create a zero array with the same dimensions but make sure it's just [frames, joints, coords]
    if len(x.shape) == 3:  # Shape is [frames, joints, coords]
        y = np.zeros(x.shape)
        
        # Map COCO keypoints to H36M
        y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5  # root = (left_hip + right_hip) / 2
        y[:,1,:] = x[:,12,:]  # right_hip
        y[:,2,:] = x[:,14,:]  # right_knee
        y[:,3,:] = x[:,16,:]  # right_ankle
        y[:,4,:] = x[:,11,:]  # left_hip
        y[:,5,:] = x[:,13,:]  # left_knee
        y[:,6,:] = x[:,15,:]  # left_ankle
        y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5  # neck = (left_shoulder + right_shoulder) / 2
        y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5  # belly = (root + neck) / 2
        y[:,9,:] = x[:,0,:]  # nose
        y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5  # head = (left_eye + right_eye) / 2
        y[:,11,:] = x[:,5,:]  # left_shoulder
        y[:,12,:] = x[:,7,:]  # left_elbow
        y[:,13,:] = x[:,9,:]  # left_wrist
        y[:,14,:] = x[:,6,:]  # right_shoulder
        y[:,15,:] = x[:,8,:]  # right_elbow
        y[:,16,:] = x[:,10,:]  # right_wrist
        
        return y
    else:
        raise ValueError(f"Expected keypoint shape [frames, joints, coords], got {x.shape}")

def generate_2d_pose(video_path, bounding_box):

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        raise SystemExit

    # Get properties of the original video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_size = (width, height)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize person detection model & processor
    person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

    # 2. Initialize pose estimation model & processor
    pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
    pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").to(device)

    frame_count = 0

    # Prepare a list to hold all frames' pose data
    all_frame_poses = []

    # Person detected flag
    person_detected = False

    # Person detected from frame number
    person_detected_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error reading

        # ------------------------------------------------------------
        # 0) Crop frame to bounding box if provided
        # ------------------------------------------------------------
        if bounding_box is not None:
            x, y, w, h = bounding_box
            frame = frame[y:y+h, x:x+w]

        # ------------------------------------------------------------
        # 1) Convert the frame (BGR) to PIL (RGB)
        # ------------------------------------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # ------------------------------------------------------------
        # 2) Detect persons
        # ------------------------------------------------------------
        inputs = person_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = person_model(**inputs)

        # Convert detector outputs to bounding boxes
        results = person_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(pil_image.height, pil_image.width)]),
            threshold=0.3  # Confidence threshold for detection
        )
        
        result = results[0]
        
        # Filter for label=0 (person) 
        person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()
        # Convert (x1, y1, x2, y2) -> (x1, y1, w, h)
        if len(person_boxes) > 0:
            person_boxes[:, 2] -= person_boxes[:, 0]
            person_boxes[:, 3] -= person_boxes[:, 1]

            # Calculate center x for each bounding box
            center_x = person_boxes[:, 0] + (person_boxes[:, 2] / 2)

            # Index of the person who is the furthest right
            rightmost_idx = np.argmax(center_x)

            # Keep only that bounding box
            person_boxes = person_boxes[rightmost_idx:rightmost_idx+1]
            
            if not person_detected:
                person_detected = True
                person_detected_frame = frame_count

        else:
            # If no person detected, add empty frame with zeros
            empty_keypoints = np.zeros((17, 3))  # 17 keypoints, x, y, confidence
            all_frame_poses.append(empty_keypoints)
            frame_count += 1
            continue  # proceed to next frame
        
        # ------------------------------------------------------------
        # 3) Pose estimation
        # ------------------------------------------------------------
        boxes_per_image = [person_boxes]
        dataset_index = torch.zeros((1, len(person_boxes)), dtype=torch.int64).to(device)  # COCO=0

        # Pose estimation
        pose_inputs = pose_processor(
            pil_image,
            boxes=boxes_per_image,
            return_tensors="pt"
        ).to(device)

        pose_inputs["dataset_index"] = dataset_index

        with torch.no_grad():
            pose_outputs = pose_model(
                pixel_values=pose_inputs["pixel_values"],
                dataset_index=pose_inputs["dataset_index"]
            )

        # Post-process pose estimation
        pose_results = pose_processor.post_process_pose_estimation(
            pose_outputs,
            boxes=boxes_per_image,
            threshold=0.1  # Confidence threshold for keypoints
        )
        
        image_pose_result = pose_results[0]

        # We'll assume only one person pose (since we keep only the rightmost person).
        # But if image_pose_result has more than one for some reason, pick the first.
        
        if len(image_pose_result) > 0:
            person_pose = image_pose_result[0]
            keypoints = person_pose["keypoints"]  # shape [num_keypoints, 2]
            scores = person_pose["scores"]        # shape [num_keypoints]
            labels = list(person_pose["labels"])
            
            # Create a keypoint array with 17 joints in COCO format
            formatted_keypoints = np.zeros((17, 3))  # x, y, confidence for 17 joints
            
            # Map the detected keypoints to the appropriate indices
            for i, label in enumerate(labels):
                if 0 <= label < 17:  # Ensure it's within our 17 keypoints
                    formatted_keypoints[label, 0] = keypoints[i][0]  # x
                    formatted_keypoints[label, 1] = keypoints[i][1]  # y
                    formatted_keypoints[label, 2] = scores[i]        # confidence
            
            all_frame_poses.append(formatted_keypoints)
        else:
            # If no pose detected, add empty frame
            empty_keypoints = np.zeros((17, 3))  # 17 keypoints, x, y, confidence
            all_frame_poses.append(empty_keypoints)

        frame_count += 1

    cap.release()
    
    # Convert to numpy array with shape [frames, 17, 3]
    all_frame_poses = np.array(all_frame_poses)
    
    # Convert from COCO to H36M format
    all_frame_poses_h36m = coco2h36m(all_frame_poses)

    return all_frame_poses_h36m, frame_count, person_detected_frame, vid_size

if __name__ == "__main__":
    video_path = "data/abc2.mp4"

    all_frame_poses, frame_count, person_detected_frame, vid_size = generate_2d_pose(video_path)
    
    # Save the keypoints to a npz file
    np.savez("data/keypoints_2d.npz", 
             keypoints=all_frame_poses, 
             frame_count=frame_count, 
             person_detected_frame=person_detected_frame,
             vid_size=vid_size)
    
    print(f"Processed {frame_count} frames")
    print(f"Person detected from frame {person_detected_frame}")
    print(f"Saved keypoints with shape {all_frame_poses.shape} in H36M format")