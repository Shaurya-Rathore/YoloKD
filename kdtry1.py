import torch
from ultralytics import YOLO
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize W&B
wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775") 
wandb.init(project="yolov8-LDConv")

# Load the custom model configuration
model = YOLO('yolov8-LDconv.yaml')

# Load the pretrained weights
# model_state_dict = torch.load('/kaggle/input/yolov8m-pt/yolov8m.pt')
# model.model.load_state_dict(model_state_dict, strict=False)
# model.model.to(device)
print(device)
# Check for valid labels
def check_labels(labels, num_classes):
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        print(f"Invalid label detected! Label values should be between 0 and {num_classes - 1}.")
    else:
        print("All labels are valid.")

# Log shapes of predictions and labels
def log_shapes(predictions, labels):
    print("Prediction shape:", predictions.shape)
    print("Label shape:", labels.shape)

# Log and check loss values
def log_losses(trainer):
    loss_items = trainer.loss_items
    print(f"Losses - Box: {loss_items[0]}, Class: {loss_items[1]}, DFL: {loss_items[2]}")
    
    if torch.isnan(loss_items[0]) or torch.isinf(loss_items[0]):
        print("Warning: Box loss contains NaN or inf values")
    if torch.isnan(loss_items[1]) or torch.isinf(loss_items[1]):
        print("Warning: Class loss contains NaN or inf values")
    if torch.isnan(loss_items[2]) or torch.isinf(loss_items[2]):
        print("Warning: DFL loss contains NaN or inf values")
    
    # wandb.log({
    #     "train/box_loss": loss_items[0],
    #     "train/cls_loss": loss_items[1],
    #     "train/dfl_loss": loss_items[2]
    # }, step=trainer.epoch)
    torch.cuda.empty_cache()

# Register the callback with the YOLO model
model.add_callback('on_train_batch_end', log_losses)

# Train the model with a reduced batch size
Result_Final_model = model.train(
    data='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/data.yaml',
    epochs=50,
    batch=8,
    optimizer='auto',
    project='yolov8-LDConv',
    save=True
)

# Save the model after training
torch.save(model.model.state_dict(), '/kaggle/working/yolov8m_custom_weights.pt')
# Finish W&B run
wandb.finish()