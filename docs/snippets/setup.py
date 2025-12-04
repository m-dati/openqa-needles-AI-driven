import os
from ultralytics import YOLO
# trained model saved here after training
MODEL_PATH = "data/yolo_project/runs/detect/openQA_element_detector_v1/weights/best.pt" 
# This file should contain our class names (0: login_button, 1: username_field, etc.)
