import cv2
import numpy as np
import json
import os
video_name = 'out2'
video_path = f"../data/videos/{video_name}.mp4"
output_path = f"../data/output/{video_name}_yolo_kalman_ball.mp4" # Output path for the final video
storage_dir = '../data/processed_frames_all_detections' # Directory to store annotated frames and all detection data



def is_round(contour, tolerance=0.1):
    """
    Checks if a contour is approximately round based on its circularity.

    Args:
        contour: The contour to check.
        tolerance: A threshold to determine how close the circularity
                   needs to be to 1. A lower value means a stricter check.

    Returns:
        True if the contour is approximately round, False otherwise.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False  # Avoid division by zero for very small contours

    area = cv2.contourArea(contour)
    if area == 0:
        return False  # Avoid division by zero for degenerate contours

    # Calculate the radius of a circle with the same perimeter
    radius = perimeter / (2 * np.pi)
    circular_area = np.pi * (radius ** 2)

    # Calculate the circularity (ratio of contour area to the area of the circle)
    circularity = area / circular_area

    # A perfect circle has a circularity of 1.
    # We use a tolerance to account for imperfections.
    return 1 - tolerance <= circularity <= 1 + tolerance


# Ensure storage directory exists
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

# Data structure to store all detection info
# Dictionary where keys are frame indices (as strings)
# Values are lists of dictionaries, each representing a detection
all_detections_data = {}

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# ... (get video properties - optional for saving, but useful later) ...

frame_idx = 0
print(f"Processing video frames and detecting all objects for storage in {storage_dir}...")

lower_hsv = np.array([100, 40, 80])   # conservative lower bound
upper_hsv = np.array([180, 160, 200]) # conservative upper bound


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame is None:
        print(f"Warning: Received None frame at index {frame_idx}")
        frame_idx += 1
        continue
    frame = cv2.resize(frame,(1600,1024))
    #dst = cv2.GaussianBlur(frame,(5,5),3)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray, threshold1=100,threshold2=200)
    
        # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(gray)
    kernel = np.ones((5,5),np.uint8)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=True)
        if 10 <= perimeter <= 1000 and is_round(cnt,tolerance=1.9):  # keep only small-edge objects
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
            
    # Display frame and wait for key press
    #cv2.imshow('Frame', dst)
    closing = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Image', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()  # Add this to clean up windows
