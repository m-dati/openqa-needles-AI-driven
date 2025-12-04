from ultralytics import YOLO
import cv2

# Load custom trained model
model = YOLO("data/yolo_project/runs/detect/openQA_element_detector_v1/weights/best.pt")

# Path to the screenshot we want to test
input_screenshot = "data/yolo_project/datasets/images/test/modified_screenshot.png"

# Run prediction
results = model.predict(
    source=input_screenshot, 
    conf=0.25, # Only show detections with confidence >= 25%
    save=True, # Save the output image with bounding boxes drawn
    show=False # Don't open a pop-up window
)

# Process the Results (Adaptation for openQA)
detections_for_openqa = []

# Loop through each detected object in the result list
for r in results:
    # 'r.boxes' contains the bounding box data
    for box in r.boxes:
        # Get the class name (e.g., 'login_button')
        class_id = int(box.cls)
        class_name = model.names[class_id]
        
        # Get the confidence score
        confidence = float(box.conf)
        
        # Get the bounding box coordinates (in pixel space)
        # xyxy: [x1, y1, x2, y2] - top-left x/y to bottom-right x/y
        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

        detections_for_openqa.append({
            "element": class_name,
            "confidence": confidence,
            "coords": [x1, y1, x2, y2]
        })

print("\n--- openQA Target Output ---")
print(detections_for_openqa)
# end
