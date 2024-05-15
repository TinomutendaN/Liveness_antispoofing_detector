import os

from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)  # 0 represents the default webcam, you can change it if you have multiple cameras
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (W, H))


model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform object detection on the frame
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw a rectangle around the detected object
        if score > 0.5:  # If the score is below 0.5, consider it fake   
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame with the object detection results
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break


cap.release()
out.release()
cv2.destroyAllWindows()