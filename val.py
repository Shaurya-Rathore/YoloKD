import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov5n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov5n-patched-1280',
              plots=True,
              )
    
    print('\n1\n')
    
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov6n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov6n-patched-1280',
              plots=True,
              )
    
    print('\n2\n')
    

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov8n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov8n-patched-1280',
              plots=True,
              )
    
    print('\n3\n')
    
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov9t_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov9t-patched-1280',
              plots=True,
              )
    
    print('\n4\n')
    
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov10n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov10n-patched-1280',
              plots=True,
              )
    
    print('\n5\n')
    
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_SSFF+loss+p2+soap_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/bucktales_patched/dtc2023_local.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='SSFF+loss+p2+soap-patched-1280',
              plots=True,
              )
    
    print('\n6\n')