import os
import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import filedialog
import os
import cv2
import numpy as np
import argparse
import glob
import csv
from ultralytics import YOLO

# tkinter is optional (only used for file dialog if no --input)
try:
    from tkinter import filedialog
    import tkinter as tk
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Load YOLO model
model = YOLO("yolov8n.pt")


def draw_detection(frame, x1, y1, x2, y2, label=None, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    if label:
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_image(path, conf_thresh=0.25, headless=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to read image:", path)
        return 0

    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else np.ones(len(boxes))
    names = results[0].names if hasattr(results[0], 'names') else {}

    person_count = 0
    for i, box in enumerate(boxes):
        if confs[i] < conf_thresh:
            continue
        x1, y1, x2, y2 = box
        cls = int(classes[i])
        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        draw_detection(img, x1, y1, x2, y2, label=f"{label} {confs[i]:.2f}")
        if cls == 0:
            person_count += 1

    # Overlay person count
    cv2.putText(img, f"People: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    base, ext = os.path.splitext(path)
    out_path = f"{base}_detected_{person_count}{ext}"
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to: {out_path} (People: {person_count})")

    if not headless:
        cv2.imshow('Detections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return person_count


def process_video(path, conf_thresh=0.25):
    cap = cv2.VideoCapture(path)
    line_position = 300
    line_thickness = 2
    direction = 'vertical'
    count_in, count_out = 0, 0
    track_ids = set()
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else np.ones(len(detections))

        # Draw counting line
        if direction == 'vertical':
            cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 255, 255), line_thickness)
        else:
            cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), line_thickness)

        for i, box in enumerate(detections):
            if confs[i] < conf_thresh:
                continue
            cls = int(classes[i])
            if cls != 0:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            draw_detection(frame, x1, y1, x2, y2, label='person')

            track_id = f"{cx}_{cy}"
            prev = track_history.get(track_id, (cx, cy))

            if direction == 'vertical':
                if prev[0] < line_position <= cx:
                    count_in += 1
                    track_ids.add(track_id)
                elif prev[0] > line_position >= cx:
                    count_out += 1
                    track_ids.add(track_id)
            else:
                if prev[1] < line_position <= cy:
                    count_in += 1
                    track_ids.add(track_id)
                elif prev[1] > line_position >= cy:
                    count_out += 1
                    track_ids.add(track_id)

            track_history[track_id] = (cx, cy)

        cv2.putText(frame, f"In: {count_in}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Out: {count_out}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Crowd Counting', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return ext in image_exts


def batch_process(dir_path, conf_thresh=0.25, out_csv=None):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.tiff", "**/*.tif", "**/*.webp"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(dir_path, p), recursive=True))
    files = sorted(files)
    if not files:
        print("No image files found in folder:", dir_path)
        return

    rows = []
    for f in files:
        count = process_image(f, conf_thresh=conf_thresh, headless=True)
        rows.append((f, count))

    if out_csv:
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['path', 'person_count'])
            writer.writerows(rows)
        print(f"Saved counts CSV to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description='Detect objects (images/videos) with YOLOv8 and count people in images.')
    parser.add_argument('--input', '-i', help='Path to image or video file')
    parser.add_argument('--batch-dir', '-b', help='Directory to batch-process images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default 0.25)')
    parser.add_argument('--headless', action='store_true', help='Run without displaying windows')
    parser.add_argument('--csv', help='Output CSV path when using --batch-dir')
    args = parser.parse_args()

    media_path = None
    if args.input:
        media_path = args.input
    elif args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print("Batch directory does not exist:", args.batch_dir)
            return
        out_csv = args.csv if args.csv else os.path.join(args.batch_dir, 'counts.csv')
        batch_process(args.batch_dir, conf_thresh=args.conf, out_csv=out_csv)
        return
    else:
        if TK_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            media_path = filedialog.askopenfilename(title='Select Image or Video File', filetypes=[('All Files', '*.*')])
            root.destroy()
        else:
            print('No input provided and tkinter not available. Use --input or --batch-dir.')
            return

    if not media_path:
        print('No file selected. Exiting...')
        return

    if is_image_file(media_path):
        print('Selected image:', media_path)
        process_image(media_path, conf_thresh=args.conf, headless=args.headless)
    else:
        print('Selected video:', media_path)
        process_video(media_path, conf_thresh=args.conf)


if __name__ == '__main__':
    main()
