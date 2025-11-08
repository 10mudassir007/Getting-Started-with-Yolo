from ultralytics import YOLO

# Load a segmentation model
model = YOLO("yolo11n-seg.pt") 

# Predict masks on an image
results = model("2.image-segmentation\\bus.jpg")

# Show results
for res in results:
    res.show()

# Get masks as numpy arrays
    masks = res.masks.data

    res.save("img-seg.jpg")