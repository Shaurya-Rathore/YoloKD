import torch
import torch.nn as nn
from ultralytics import YOLO
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import yaml

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SoftSharedYOLO(nn.Module):
    def __init__(self, base_model: YOLO, bank_size: int = 2):
        super().__init__()
        self.base_model = base_model
        self.bank_size = bank_size
        
        # Initialize filter banks for convolution layers
        self.filter_banks = nn.ModuleDict()
        self.coefficients = nn.ParameterDict()
        
        # Create filter banks for each convolution layer
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                bank = nn.ModuleList([
                    nn.Conv2d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        bias=module.bias is not None
                    ) for _ in range(bank_size)
                ])
                self.filter_banks[name] = bank
                
                # Initialize sharing coefficients
                coeff = nn.Parameter(torch.ones(bank_size) / bank_size)
                self.coefficients[name] = coeff

    def forward(self, x):
        # Store intermediate outputs for each filter bank
        layer_outputs = {}
        
        # Process through each layer with soft sharing
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                bank_outputs = []
                for i in range(self.bank_size):
                    bank_output = self.filter_banks[name][i](x)
                    bank_outputs.append(bank_output)
                
                # Combine outputs using learned coefficients
                coeffs = torch.softmax(self.coefficients[name], dim=0)
                x = sum(c * out for c, out in zip(coeffs, bank_outputs))
            else:
                x = module(x)
                
        return x

class EnhancedYOLOTrainer:
    def __init__(
        self,
        data_yaml_path: str,
        model_type: str = 'yolov8m.yaml',
        project_name: str = 'yolov8_softshare',
        wandb_key: str = None,
        bank_size: int = 2,
        device: str = None,
        schedule: List[int] = [60, 120, 160],
        gammas: List[float] = [0.2, 0.2, 0.2],
        momentum: float = 0.9,
        weight_decay: float = 0.0005
    ):
        """
        Initialize Enhanced YOLO trainer with soft sharing and W&B integration
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.project_name = project_name
        self.wandb_key = wandb_key
        self.bank_size = bank_size
        self.schedule = schedule
        self.gammas = gammas
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Initialize model with soft sharing
        try:
            base_model = YOLO('yolov8m.yaml')
            self.model = SoftSharedYOLO(base_model, bank_size=bank_size)
            self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"Failed to initialize Soft Shared YOLO model: {e}")
            raise
            
        # Initialize metrics tracking
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.best_val_loss = float('inf')
        
    def setup_optimizer(self, learning_rate: float):
        """Configure optimizer with parameter groups for soft sharing"""
        # Group parameters - separate weight decay for coefficients
        params = [
            {'params': [p for n, p in self.model.named_parameters() if 'coefficients' not in n],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'coefficients' in n],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = SGD(params, lr=learning_rate, momentum=self.momentum)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.schedule, gamma=self.gammas[0])

    def setup_wandb(self) -> None:
        """Initialize W&B logging with soft sharing parameters"""
        try:
            if self.wandb_key:
                wandb.login(key=self.wandb_key)
            
            config = {
                'bank_size': self.bank_size,
                'schedule': self.schedule,
                'gammas': self.gammas,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay
            }
            
            wandb.init(project=self.project_name, config=config)
            
        except Exception as e:
            logging.error(f"Failed to initialize W&B: {e}")
            raise

    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Enhanced logging including soft sharing coefficients"""
        try:
            # Log basic metrics
            wandb.log(metrics, step=step)
            
            # Log coefficient distributions
            for name, coeff in self.model.coefficients.items():
                wandb.log({
                    f'coefficients/{name}': wandb.Histogram(coeff.detach().cpu().numpy())
                }, step=step)
                
        except Exception as e:
            logging.warning(f"Failed to log metrics to W&B: {e}")

    def save_checkpoint(self, epoch: int, is_best: bool, save_dir: str):
        """Save model checkpoint with soft sharing state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path / 'checkpoint.pth')
        if is_best:
            torch.save(checkpoint, save_path / 'model_best.pth')

    def train(
        self,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 0.1,
        save_dir: str = None
    ) -> Dict[str, Any]:
        """
        Train the Soft Shared YOLO model
        """
        self.setup_optimizer(learning_rate)
        
        try:
            # Configure training parameters
            train_args = {
                'data': str(self.data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'project': self.project_name,
                'device': [self.device],
            }
            
            if save_dir:
                train_args['project'] = save_dir
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                epoch_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = self.validate()
                
                # Update learning rate
                self.scheduler.step()
                
                # Save checkpoint
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                if save_dir:
                    self.save_checkpoint(epoch, is_best, save_dir)
                
                # Log metrics
                combined_metrics = {**epoch_metrics, **val_metrics}
                self.log_metrics(epoch, epoch * len(self.train_loader), combined_metrics)
            
            return {'best_val_loss': self.best_val_loss}
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
        
        finally:
            wandb.finish()
            torch.cuda.empty_cache()

def main():
    trainer = EnhancedYOLOTrainer(
        data_yaml_path='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
        wandb_key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775",
        bank_size=2,
        schedule=[60, 120, 160],
        gammas=[0.2, 0.2, 0.2],
        momentum=0.9,
        weight_decay=0.0005
    )
    
    trainer.setup_wandb()
    
    try:
        results = trainer.train(
            epochs=40,
            batch_size=8,
            learning_rate=0.1,
            save_dir='yolov8_softshare'
        )
        logging.info(f"Training completed successfully: {results}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()