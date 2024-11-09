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
            self.model.to(self.device)
            # Move model to device
            if hasattr(self.model, 'model'):
                self.model.model = self.model.model.to(self.device)
                
            # Ensure any internal models/modules are on the correct device
            if hasattr(self.model, 'trainer') and hasattr(self.model.trainer, 'model'):
                self.model.trainer.model = self.model.trainer.model.to(self.device)
                
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
                'device': [self.device],  # YOLOv8 expects device as a list
            }
            
            if save_dir:
                train_args['project'] = save_dir
                
            # Start training
            logger.info(f"Starting training on device: {self.device}")
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
        """
        Validate the model on a validation dataset
        """
        self.model.eval()
        
        try:
            with torch.no_grad():
                results = self.model.val(**{
                    'data': str(self.data_yaml_path),
                    'device': [self.device]
                })
                return results
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
            
        finally:
            self.model.train()

def main():
    # Parse command line arguments if needed
    trainer = YOLOTrainer(
        data_yaml_path='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/data.yaml',
        wandb_key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775"
    )
    
    trainer.setup_wandb()
    
    try:
        results = trainer.train(
            epochs=35,
            batch_size=8,
            save_dir='yolov8_paramshare'
        )
        logger.info(f"Training completed successfully: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()