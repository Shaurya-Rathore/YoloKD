import torch
from ultralytics import YOLO
import wandb
from pathlib import Path
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
        project_name: str = 'yolov8',
        wandb_key: str = None,
        device: str = None
    ):
        self.data_yaml_path = Path(data_yaml_path)
        self.project_name = project_name
        self.wandb_key = wandb_key
        
        # Validate data yaml exists
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data YAML file not found at {data_yaml_path}")
            
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        try:
            self.model = YOLO(model_type)
            self.model.model.to(self.device)
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
            wandb.log({
                "train/box_loss": loss_items[0],
                "train/cls_loss": loss_items[1],
                "train/dfl_loss": loss_items[2]
            }, step=trainer.epoch)
        except Exception as e:
            logger.warning(f"Failed to log losses to W&B: {e}")

    def train(
        self,
        epochs: int = 35,
        batch_size: int = 8,
        optimizer: str = 'auto',
        save_dir: str = None
    ) -> Dict[str, Any]:

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
                'save': True
            }
            
            if save_dir:
                train_args['project'] = save_dir
                
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
        
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        try:
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    predictions = self.model(images)
                    # Add validation metric logging here
                    
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
            
        finally:
            self.model.train()

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = YOLOTrainer(
        data_yaml_path='/path/to/data.yaml',
        wandb_key="your-wandb-key"
    )
    
    # Setup W&B logging
    trainer.setup_wandb()
    
    # Start training
    results = trainer.train(
        epochs=35,
        batch_size=8,
        save_dir='yolov8_training'
    )