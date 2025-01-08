import wandb
from ultralytics import YOLO
# from ultralytics.nn.modules.darts_utils import AvgrageMeter, process_yolov8_output, accuracy
from torchsummary import summary
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Load the test data
data_path = '/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test'

# # Initialize Wandb
# wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775")
# wandb.init(project="yolov8-inference")

# Load the model with custom weights
model_path_1 = "/kaggle/input/best.pt"
model_path_2 = "/kaggle/input/yolov8_softshare_waid.pt"

model = YOLO(model_path_1)
# Run batched inference on the dataset specified in the YAML file
results = model(data_path)

# Select 500 random indices from the results
random_indices = random.sample(range(len(results)), 500)

# Process and save 500 random results with unique filenames
for idx in random_indices:
    result = results[idx]
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename=f"best_images/best_{idx}.jpg")  # save to disk with unique filename


model = YOLO(model_path_2)
# Run batched inference on the dataset specified in the YAML file 
results = model(data_path)

# Select 500 random indices from the results
random_indices = random.sample(range(len(results)), 500)

# Process and save 500 random results with unique filenames
for idx in random_indices:
    result = results[idx]
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename=f"softshare_images/softshare_{idx}.jpg")  # save to disk with unique filename

    
image1 = mpimg.imread('/kaggle/working/best.jpg')
image2 = mpimg.imread('/kaggle/working/softshare.jpg')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 20))

axes[0].imshow(image1)
axes[0].set_title('SPD')
axes[0].axis('off')

# Display the second image
axes[1].imshow(image2)
axes[1].set_title('Simple')
axes[1].axis('off')  # Hide the axis

# Save the comparision plot
plt.savefig('compare.jpg', dpi=300)

plt.show()

# # Finish the Wandb run
# wandb.finish()
