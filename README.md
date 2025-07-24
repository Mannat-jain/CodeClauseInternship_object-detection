# CodeClauseInternship_object-detection
🎯 Built using MobileNet SSD & YOLOv5

🖼️ Detects objects in static images & real-time webcam feed

💻 Tech Stack: Python, OpenCV, PyTorch, Tkinter
_________________________________________________________________________________________

💡 What This Project Does

This project detects and classifies objects using deep learning models:
_________________________________________________________________________________________

🖼️ Image Detection

→ Implemented with MobileNet SSD and YOLOv5
→ YOLOv5 offers better accuracy, MobileNet works faster on CPU

🎥 Webcam Detection

→ Built with MobileNet SSD only (so far)
→ Future version may include YOLO and even video file detection
_________________________________________________________________________________________

🔍 Models Used

🔹 YOLOv5 (You Only Look Once)

A fast and accurate object detection algorithm.
Used here for static image detection.

📦 Based on PyTorch

🧠 Great for multi-object recognition

🔬 Accurate, but requires GPU for real-time performance

🐌 A bit slower on my system (CPU-only)

🔹 MobileNet SSD

A lightweight model perfect for lower-resource environments.
Used for both image & webcam detection.

⚡ Super fast on CPU

📱 Designed for mobile and edge devices

🪶 Smaller in size, easy to implement
_________________________________________________________________________________________

⚖️ YOLOv5 vs MobileNet SSD

Feature	     |              YOLOv5  	    |        MobileNet SSD

Accuracy	     |           ✅ High	            |  ➖ Moderate

Speed on CPU	   |         ❌ Slower	        |    ✅ Faster

Speed on GPU	  |          ✅ Real-time	    |      ✅ Real-time

File Size	        |          Larger	      |          Smaller

Best For	          |        Images	    |          Real-Time/Edge

On my i5 laptop (CPU-only), MobileNet runs smoothly in real-time.

YOLOv5 works best for static images due to load time.
_________________________________________________________________________________________

🚀 What's Unique

✔️ Real-time webcam detection (MobileNet)

✔️ Static image detection (YOLOv5 & MobileNet)

📂 Saves detected output with bounding boxes

🧪 Using only CPU for now (performance may vary)

🛠️ Plan to add YOLO to video detection soon!
_________________________________________________________________________________________

🛠️ Technologies Used

Python

OpenCV

PyTorch (YOLOv5)

Caffe (MobileNet SSD)

Tkinter (for image selector)
_________________________________________________________________________________________

🔮 What’s Next?

Add YOLOv5 to live video stream detection

Try GPU acceleration for faster processing

Expand supported object classes or train a custom model

Possibly build a web app version
_________________________________________________________________________________________
🎥 Project Demo

This game was built with an intuitive interface and real-time interaction. You can watch a quick demo of it in action on my Linkedin.
_________________________________________________________________________________________

🙋‍♀️ Built by:

Mannat Jain

Computer Science Student | AI & Python Enthusiast

https://www.linkedin.com/in/mannatjain14/ | https://github.com/Mannat-jain
