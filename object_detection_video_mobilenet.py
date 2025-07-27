import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and class labels
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Class labels from the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Assign random colors for each class for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Start video stream from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fixed size and convert to blob
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    # Pass blob to the network and perform forward pass
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only show detections above 50% confidence
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Get object box coordinates
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                       frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            label_text = f"{label}: {confidence * 100:.1f}%"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Object Detection", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
