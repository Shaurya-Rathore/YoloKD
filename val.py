import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov5n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=2560,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov5n-vanilla-unpatched-2560',
              plots=True,
              )
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov6n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=2560,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov6n-vanilla-unpatched-2560',
              plots=True,
              )
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov9t_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
#               split='test',
#               imgsz=2560,
#               batch=16,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov9t-vanilla-unpatched-2560',
#               plots=True,
#               )
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov10n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=2560,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov10n-vanilla-unpatched-2560',
              plots=True,
              )