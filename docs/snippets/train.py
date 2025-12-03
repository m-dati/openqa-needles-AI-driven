from ultralytics import YOLO
# 1. Load a pre-trained model (recommended for custom data)
# 'n' for nano is the smallest and fastest model—great for a POC.
model = YOLO("yolov8n.pt")

# 2. Train the model
# 'data': path to our YAML configuration file
# 'epochs': number of training cycles (start low, then increase)
# 'imgsz': input image size (640x640 is standard)
# 'batch': number of images per training step (adjust based on our GPU/CPU memory)

results = model.train(
data="training_config.yaml",
epochs=50,
imgsz=640,
batch=8,
name="openQA_element_detector_v1" # Name for our training run
directory
)

# The best weights will be saved to: 
# data/yolo_project/runs/detect/openQA_element_detector_v1/weights/best.pt
print("Training complete! Model saved.")
# end