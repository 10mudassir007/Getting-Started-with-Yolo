from ultralytics import YOLO
import cv2

# Load a segmentation model
model = YOLO("yolo11n-seg.pt") 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run segmentation
    results = model(frame)[0]

    # results.masks contains the segmentation masks
    # results.boxes contains bounding boxes (optional)
    # Overlay masks on the frame
    annotated_frame = results.plot()  # YOLOv8 built-in plotting

    cv2.imshow("YOLOv12 Live Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()