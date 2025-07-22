import cv2
import os
import random

def get_width(video_path):
    """
    Get the width of a video file.
    
    Args:
        video_path (str): Absolute path to the video file
        
    Returns:
        int: Width of the video in pixels
        
    Raises:
        ValueError: If the video file cannot be opened or read
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Error: Could not open video file '{video_path}'")
    
    # Get the width property
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Release the video capture object
    cap.release()
    
    return width

def get_bounding_box(video_url):
    """
    Get bounding box coordinates for the right half of a video.
    
    Args:
        video_url (str): URL or path to the video file
        
    Returns:
        tuple: (x, y, width, height) coordinates of the bounding box for the right half
               Returns None if video width is not 1280 or 3840
        
    Raises:
        ValueError: If the video file cannot be opened or read
    """
    # Get the width of the video
    width = get_width(video_url)
    
    # Check if width is 1280 or 3840
    if width != 1280 and width != 3840:
        return None
    
    # Open the video to get height
    cap = cv2.VideoCapture(video_url)
    
    # Check if video opened successfully
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Error: Could not open video file '{video_url}'")
    
    # Get the height property
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate bounding box based on width
    if width == 1280:
        # For 1280 width: use right half
        x = width // 2  # Start from middle (right half)
        y = 0           # Start from top
        box_width = width // 2  # Half width
        box_height = height     # Full height
    elif width == 3840:
        # For 3840 width: analyze 2nd and 4th pieces to find darker one
        piece_width = width // 4  # Width of each piece
        
        # Get a sample frame to analyze
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)  # Use frame 1000 for analysis
        ret, frame = cap.read()
        
        if ret:
            # Extract 2nd piece (x = piece_width to 2*piece_width)
            piece_2_x = piece_width
            piece_2 = frame[0:height, piece_2_x:piece_2_x + piece_width]
            
            # Extract 4th piece (x = 3*piece_width to 4*piece_width)
            piece_4_x = 3 * piece_width
            piece_4 = frame[0:height, piece_4_x:piece_4_x + piece_width]
            
            # Calculate average brightness for each piece
            # Convert to grayscale for brightness analysis
            piece_2_gray = cv2.cvtColor(piece_2, cv2.COLOR_BGR2GRAY)
            piece_4_gray = cv2.cvtColor(piece_4, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness (lower value = darker)
            brightness_2 = cv2.mean(piece_2_gray)[0]
            brightness_4 = cv2.mean(piece_4_gray)[0]
            
            # Choose the darker piece
            if brightness_2 < brightness_4:
                # 2nd piece is darker
                x = piece_2_x
                print(f"Using 2nd piece (darker): brightness_2={brightness_2:.2f}, brightness_4={brightness_4:.2f}")
            else:
                # 4th piece is darker
                x = piece_4_x
                print(f"Using 4th piece (darker): brightness_2={brightness_2:.2f}, brightness_4={brightness_4:.2f}")
        else:
            # Fallback to 4th piece if frame reading fails
            x = 3 * piece_width
            print("Frame reading failed, using 4th piece as fallback")
        
        y = 0                     # Start from top
        box_width = piece_width   # Width of one piece
        box_height = height       # Full height
    
    # Release the video capture object
    cap.release()
    
    return (x, y, box_width, box_height)

def save_frame(video_path):
    """
    Save 5 random cropped frames from a video using bounding box coordinates.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        list: List of paths to the saved frame images
        
    Raises:
        ValueError: If the video file cannot be opened or read
        ValueError: If video width is not 1080 pixels
    """
    # Get bounding box coordinates
    bounding_box = get_bounding_box(video_path)
    
    if bounding_box is None:
        return None
    
    x, y, box_width, box_height = bounding_box
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Error: Could not open video file '{video_path}'")
    
    # Get video name without extension for folder creation
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create test_frames directory if it doesn't exist
    test_frames_dir = "test_frames"
    if not os.path.exists(test_frames_dir):
        os.makedirs(test_frames_dir)

    frame_num = 1000

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Crop the frame using bounding box coordinates
        cropped_frame = frame[y:y+box_height, x:x+box_width]
        
        # Save the cropped frame
        frame_filename = f"{video_name}_{frame_num}.jpg"
        
        cv2.imwrite(test_frames_dir + "/" + frame_filename, cropped_frame)
    else:
        print(f"Warning: Could not read frame {frame_num}")
    
    # Release the video capture object
    cap.release()
    
    return 1
