import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov5n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/valt',
              name='yolov5n-vanilla-patched-640',
              plots=True,
              )
    
#     print('\n1\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov6n_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
#               split='test',
#               imgsz=640,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov6n-vanilla-patched-640',
#               plots=True,
#               )
    
#     print('\n2\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov9t_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
#               split='test',
#               imgsz=640,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov9t-vanilla-patched-640',
#               plots=True,
#               )
    
#     print('\n3\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov10n_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
#               split='test',
#               imgsz=640,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov10n-vanilla-patched-640',
#               plots=True,
#               )
    
    print('\n4\n')

# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov5n_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
#               split='test',
#               imgsz=3840,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov5n-vanilla-unpatched-3840',
#               plots=True,
#               )
    
#     print('\n\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov6n_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
#               split='test',
#               imgsz=3840,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov6n-vanilla-unpatched-3840',
#               plots=True,
#               )
    
#     print('\n\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov9t_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
#               split='test',
#               imgsz=3840,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov9t-vanilla-unpatched-3840',
#               plots=True,
#               )
    
#     print('\n\n')
    
# if __name__ == '__main__':
#     model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov10n_bucktales_patched.pt')
#     model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
#               split='test',
#               imgsz=3840,
#               batch=4,
#               # rect=False,
#               # save_json=True, # if you need to cal coco metrics
#               project='runs/val',
#               name='yolov10n-vanilla-unpatched-3840',
#               plots=True,
#               )
    
#     print('\n\n')