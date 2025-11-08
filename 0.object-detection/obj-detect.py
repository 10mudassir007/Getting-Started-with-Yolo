from ultralytics import YOLO

# Load a pretrained YOLOv8 model (you can choose v8n, v8s, v8m, etc.)
model = YOLO('yolo12n.pt')

results = model('bus.jpg')  # Replace with your image path

for r in results:
    r.show()        # Display image with detections
    r.save()

