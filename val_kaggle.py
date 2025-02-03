import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('benchmark_weights/best_yolov5n_bucktales_patched.pt')
#     model.val(data='benchmark_weights/dtc2023.yaml',
#               split='test',
#               imgsz=2560,
#               batch=16,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov5n-vanilla-unpatched-2560',
#               plots=True,
#               )
    
# if __name__ == '__main__':
#     model = YOLO('benchmark_weights/best_yolov6n_bucktales_patched.pt')
#     model.val(data='benchmark_weights/dtc2023.yaml',
#               split='test',
#               imgsz=2560,
#               batch=16,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov6n-vanilla-unpatched-2560',
#               plots=True,
#               )
    
# if __name__ == '__main__':
#     model = YOLO('benchmark_weights/best_yolo9t_bucktales_patched.pt')
#     model.val(data='benchmark_weights/dtc2023.yaml',
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
    model = YOLO('benchmark_weights/best_yolov10n_bucktales_patched.pt')
    model.val(data='benchmark_weights/dtc2023.yaml',
              split='test',
              imgsz=2560,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov10n-vanilla-unpatched-2560',
              plots=True,
              )