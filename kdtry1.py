import torch
from ultralytics import YOLO
import wandb

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize a new W&B run
wandb.login()  # Ensure the API key is set via environment or W&B CLI
wandb.init(project="yolov8-LDConv")

# Load the custom model configuration
model = YOLO('yolov8-LDconv.yaml')

# Load the pretrained model from the provided path
model_state_dict = torch.load('/kaggle/input/yolov8m-pt/yolov8m.pt')  

# Load the filtered state_dict into your model
model.model.load_state_dict(model_state_dict, strict=False)
model.model.to(device)

# Define a callback to log losses at the end of each training batch
def log_losses(trainer):
    # Access the loss dictionary
    loss_items = trainer.loss_items
    
    # Log each loss component
    wandb.log({
        "train/box_loss": loss_items[0],
        "train/cls_loss": loss_items[1],
        "train/dfl_loss": loss_items[2]
    }, step=trainer.epoch)

# Register the callback with the YOLO model
model.add_callback('on_train_batch_end', log_losses)

# Train the model with the specified configuration and sync to W&B
Result_Final_model = model.train(
    data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
    epochs=3,
    batch=16,
    optimizer='auto',
    project='yolov8-LDConv',
    save=True,
)

# Finish the W&B run
wandb.finish()



