import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov8n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
              split='test',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov8n-vanilla-patched-640',
              plots=True,
              )
    
    print('\n1\n')
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/last_SSFF+wio+nwu+soap_bucktales_patched_57.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
              split='test',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='last_SSFF+wio+nwu+soap_bucktales_patched_57-640',
              plots=True,
              )
    
    print('\n2\n')