from ultralytics import YOLO              # Import YOLO model from ultralytics
import cv2                                # OpenCV for image processing
import matplotlib.pyplot as plt         
import os
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained YOLOv5 model
model = YOLO("yolov5s.pt")                # 's' = small version, fast and lightweight

# Load your image
root = tk.Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not image_path:
    print("No file selected. Exiting.")
    exit()
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))

# Run inference (detection)
results = model(image)                   

# Draw boxes and labels on the image
results[0].plot()                        
output_path = "detected_output.jpg"       
cv2.imwrite(output_path, results[0].plot())  # Save the image with detections

# Optional: show in a pop-up (matplotlib)
img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("YOLOv5 Detection Output")
plt.axis("off")
plt.show()
