
import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(
        self,
        data_yaml_path: str,
        model_type: str = 'yolov8m.yaml',
        project_name: str = 'yolov8_paramshare',
        wandb_key: str = None,
        device: str = None
    ):
        """
        Initialize YOLO trainer with W&B integration
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.project_name = project_name
        self.wandb_key = wandb_key
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        try:
            self.model = YOLO(model_type)
            # Move entire model to specified device
            self.model = self.model.to(self.device)
            
            # Ensure criterion and other model components are on the correct device
            if hasattr(self.model, 'criterion'):
                self.model.criterion = self.model.criterion.to(self.device)
                
            # Set default device for model computations
            self.model.args.device = self.device
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def setup_wandb(self) -> None:
        """Initialize W&B logging"""
        try:
            if self.wandb_key:
                wandb.login(key=self.wandb_key)
            wandb.init(project=self.project_name)
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise

    def log_losses(self, trainer) -> None:
        """Callback to log losses to W&B after each training batch"""
        try:
            loss_items = trainer.loss_items
            # Ensure loss items are detached and moved to CPU before logging
            losses = {
                "train/box_loss": float(loss_items[0].detach().cpu()),
                "train/cls_loss": float(loss_items[1].detach().cpu()),
                "train/dfl_loss": float(loss_items[2].detach().cpu())
            }
            wandb.log(losses, step=trainer.epoch)
        except Exception as e:
            logger.warning(f"Failed to log losses to W&B: {e}")

    def train(
        self,
        epochs: int = 35,
        batch_size: int = 8,
        optimizer: str = 'auto',
        save_dir: str = None
    ) -> Dict[str, Any]:
        """
        Train the YOLO model
        """
        try:
            # Register W&B logging callback
            self.model.add_callback('on_train_batch_end', self.log_losses)
            
            # Configure training parameters
            train_args = {
                'data': str(self.data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'optimizer': optimizer,
                'project': self.project_name,
                'save': True,
                'device': self.device  # Explicitly set device for training
            }
            
            if save_dir:
                train_args['project'] = save_dir
                
            # Monkey patch the loss computation to ensure device consistency
            def loss_wrapper(criterion, preds, batch):
                # Move batch to the correct device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device)
                
                return criterion(preds, batch)
            
            if hasattr(self.model, 'criterion'):
                original_call = self.model.criterion.__call__
                self.model.criterion.__call__ = lambda x, y: loss_wrapper(original_call, x, y)
            
            # Start training
            logger.info("Starting training...")
            results = self.model.train(**train_args)
            
            # Clean up
            torch.cuda.empty_cache()
            wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            wandb.finish()
            raise

# Example usage
if __name__ == "__main__":
    trainer = YOLOTrainer(
        data_yaml_path='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/data.yaml',
        wandb_key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775"
    )
    
    trainer.setup_wandb()
    
    results = trainer.train(
        epochs=35,
        batch_size=8,
        save_dir='yolov8_paramshare'
    )