
def find_element_coordinates(screenshot_path: str, target_class_name: str, confidence_threshold: float = 0.75):
    """
    Runs YOLOv8 inference on a screenshot to find a specific element.

    :param screenshot_path: Path to the input image file (the SUT screen).
    :param target_class_name: The name of the element to find (e.g., 'login_button').
    :param confidence_threshold: Minimum confidence score to consider a detection valid.
    :return: A list of detection dictionaries, or an empty list.
    """

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return []

    # 1. Load the Trained Model
    model = YOLO(MODEL_PATH)

    # 2. Run Inference
    # The 'results' object contains all the detected bounding boxes, classes, and confidences.
    results = model.predict(source=screenshot_path, conf=confidence_threshold, verbose=False)

    detections = []
    
    # 3. Process the Results
    # YOLOv8 returns a list of Results objects (one per image/frame)
    for r in results:
        # r.boxes is the detection object containing all bounding boxes
        for box in r.boxes:
            # Get data
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            # Get the coordinates in pixel space (xyxy format: top-left x/y, bottom-right x/y)
            # The .tolist() is necessary to convert PyTorch tensors to standard Python lists
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

            detected_name = model.names[class_id] # Convert ID back to name (e.g., 0 -> 'login_button')
            
            if detected_name == target_class_name:
                detections.append({
                    "element": detected_name,
                    "confidence": round(confidence, 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

    return detections
# end
