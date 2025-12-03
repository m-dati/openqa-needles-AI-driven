from ultralytics import YOLO

model = YOLO(MODEL_PATH)
# Run validation on the test dataset (using our training_config.yaml)
metrics = model.val(data='training_config.yaml')

print(f"mAP@50 (Good Match): {metrics.results_dict['metrics/mAP50']:.4f}") 
print(f"mAP@50:95 (Strict Match): {metrics.results_dict['metrics/mAP50-95']:.4f}")
# stop
