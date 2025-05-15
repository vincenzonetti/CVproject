import cv2
from ultralytics import YOLO
model = YOLO("yolo11s_ft.pt")

video_path = '../data/videos/out2.mp4'


cap = cv2.VideoCapture(video_path)
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'avc1' for AVI or 'mp4v' for MP4
output_path = "../output/out2_yolo11ftbs.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, line_width=1, tracker="botsort.yaml")
    annotated_frame = results[0].plot()
    
    # Write the annotated frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()
