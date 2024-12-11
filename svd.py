from ultralytics.models.yolo import YOLO
import yaml
from ultralytics.cfg import v8
# Load the YAML file
with open('yolov8-softshare.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Pretty print the configuration
import pprint
pprint.pprint(config)
