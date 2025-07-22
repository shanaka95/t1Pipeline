import cv2

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
