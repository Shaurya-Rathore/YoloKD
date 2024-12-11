from ultralytics.models.yolo import YOLO

try:
    model = YOLO('yolov8-softshare.yaml', verbose=True)
    print("YAML Configuration Loaded Successfully")
except Exception as e:
    print(f"Error in YAML Configuration: {e}")