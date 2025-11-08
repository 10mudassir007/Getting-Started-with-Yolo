from ultralytics import YOLO
import cv2
# Load a pretrained YOLOv8 model (you can choose v8n, v8s, v8m, etc.)
model = YOLO('yolov8n.pt').to('vulkan')

cap = cv2.VideoCapture(0)  # 0 = default camera, change if needed

if not cap.isOpened():
    raise Exception("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Loop through results and display
    for r in results:
        annotated_frame = r.plot()  # Draw boxes and labels
        cv2.imshow("YOLOv8 Live", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()