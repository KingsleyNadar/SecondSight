from pathlib import Path
import cv2
import torch
from matplotlib import pyplot as plt

# Set the path to the YOLOv5 model weights file
model_weights_path = "yolov8n.pt"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s', pretrained=False)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

# Load an image for object detection

img = cv2.VideoCapture(0)

# Perform object detection
results = model(img)

# Display the detected objects and their bounding boxes
results.show()

# Access the detected object labels, confidence scores, and bounding box coordinates
labels = results.names
pred_labels = results.pred[0][:, -1].cpu().numpy()  # Object labels
pred_scores = results.pred[0][:, 4].cpu().numpy()  # Confidence scores
pred_boxes = results.pred[0][:, :4].cpu().numpy()  # Bounding boxes

# Iterate over detected objects and print their labels, scores, and coordinates
for label, score, box in zip(pred_labels, pred_scores, pred_boxes):
    label_str = labels[int(label)]
    print(f"Label: {label_str}, Confidence: {score:.2f}")
    print(f"Bounding Box: {box}")

# Save the results to an output image
results.save(Path("output_directory"))

# Show the image with detected objects (optional)
results.render()
plt.show()
