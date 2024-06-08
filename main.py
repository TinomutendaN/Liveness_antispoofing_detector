import cv2
import os
from ultralytics import YOLO

# Open the webcam, 0 represents the default webcam. Change if you have multiple cameras
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame from the webcam
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

# Get the frame dimensions
H, W, _ = frame.shape

# Initialize a video writer object to save the output video
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (W, H))

# Define the path to the YOLO model weights
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')

# Load a YOLO model with the custom weights
model = YOLO(model_path)

# Set a threshold for detection confidence
threshold = 0.5

# Main loop for reading frames from the webcam and performing object detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform object detection on the frame
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw a rectangle around the detected object if confidence score is above the threshold
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, "Face Detected", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame with the object detection results
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and video writer objects and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()