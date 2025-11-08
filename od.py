import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import filedialog
import tkinter as tk

# Create a root window but hide it
root = tk.Tk()
root.withdraw()

# Load the YOLOv8 model (download "yolov8n.pt" or "yolov8x.pt" if not present)
model = YOLO("yolov8n.pt")

# Open file dialog to select video file
print("Please select any video file...")
video_path = filedialog.askopenfilename(
    title="Select Any Video File",
    filetypes=[("All Video Files", ".")]
)

if not video_path:
    print("No video file selected. Exiting...")
    exit()

print(f"Selected video: {video_path}")
cap = cv2.VideoCapture(video_path)

# Define counting line coordinates (vertical or horizontal as required)
line_position = 300  # x or y coordinate of the line
line_thickness = 2
direction = 'vertical'  # or 'horizontal'
count_in, count_out = 0, 0
track_ids = set()
track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy().astype(int)
    persons = [box for i, box in enumerate(detections)
               if int(results[0].boxes.cls[i]) == 0]  # class 0 for person

    # Draw the counting line
    if direction == 'vertical':
        cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 255, 255), line_thickness)
    else:
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), line_thickness)

    for box in persons:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        track_id = f"{cx}{cy}"
        prev = track_history.get(track_id, (cx, cy))

        # Check for crossing the line
        if direction == 'vertical':
            if prev[0] < line_position <= cx:  # left to right (IN)
                count_in += 1
                track_ids.add(track_id)
            elif prev[0] > line_position >= cx:  # right to left (OUT)
                count_out += 1
                track_ids.add(track_id)
        else:
            if prev[1] < line_position <= cy:  # top to bottom (IN)
                count_in += 1
                track_ids.add(track_id)
            elif prev[1] > line_position >= cy:  # bottom to top (OUT)
                count_out += 1
                track_ids.add(track_id)

        track_history[track_id] = (cx, cy)

    # Display the counts
    cv2.putText(frame, f"In: {count_in}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Out: {count_out}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Crowd Counting', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()