
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained MobileNet SSD model and class labels
prototxt_path = r"C:\\Users\\acer\\OneDrive\\Documents\\code files\\CodeClause\\object detection\\MobileNetSSD_deploy.prototxt.txt"
model_path = r"C:\\Users\\acer\\OneDrive\\Documents\\code files\\CodeClause\\object detection\\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define class labels MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Assign random colors to each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Use Tkinter to select an image file
root = tk.Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not image_path:
    print("No file selected. Exiting.")
    exit()

image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Convert the image to a blob for input into the network
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                             0.007843, (300, 300), 127.5)

# Feed the input to the network
net.setInput(blob)
detections = net.forward()

# Loop through detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx]

        # Get bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the prediction on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        text = f"{label}: {confidence*100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Display the output image
cv2.imshow("Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
