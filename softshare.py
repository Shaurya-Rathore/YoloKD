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

class SoftSharedYOLO(nn.Module):
    def __init__(self, base_model: YOLO, bank_size: int = 2):
        super().__init__()
        self.base_model = base_model
        self.bank_size = bank_size
        
        # Create mappings to store layer references
        self.layer_mapping = {}
        self.filter_banks = nn.ModuleDict()
        self.coefficients = nn.ParameterDict()
        
        # Create filter banks for each convolution layer
        layer_count = 0
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Use a simple numeric identifier instead of the full path
                layer_id = f'conv_{layer_count}'
                self.layer_mapping[name] = layer_id
                
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
                self.filter_banks[layer_id] = bank
                
                # Initialize sharing coefficients
                coeff = nn.Parameter(torch.ones(bank_size) / bank_size)
                self.coefficients[layer_id] = coeff
                
                layer_count += 1

    def forward(self, x):
        # Process through each layer with soft sharing
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_id = self.layer_mapping[name]
                bank_outputs = []
                
                for i in range(self.bank_size):
                    bank_output = self.filter_banks[layer_id][i](x)
                    bank_outputs.append(bank_output)
                
                # Combine outputs using learned coefficients
                coeffs = torch.softmax(self.coefficients[layer_id], dim=0)
                x = sum(c * out for c, out in zip(coeffs, bank_outputs))
            else:
                x = module(x)
                
        return x

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
            base_model = YOLO(model_type)
            self.model = SoftSharedYOLO(base_model, bank_size=bank_size)
            self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"Failed to initialize Soft Shared YOLO model: {e}")
            raise
            
        # Initialize metrics tracking
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.best_val_loss = float('inf')

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

    def setup_optimizer(self, learning_rate: float):
        """Configure optimizer with parameter groups for soft sharing"""
        params = [
            {'params': [p for n, p in self.model.named_parameters() if 'coefficients' not in n],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'coefficients' in n],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = SGD(params, lr=learning_rate, momentum=self.momentum)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.schedule, gamma=self.gammas[0])

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
            # Configure training parameters using YOLO's training interface
            train_args = {
                'data': str(self.data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'project': self.project_name if not save_dir else save_dir,
                'device': [self.device],
            }
            
            # Start training using YOLO's training method
            results = self.model.base_model.train(**train_args)
            
            return results
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
        
        finally:
            wandb.finish()
            torch.cuda.empty_cache()

    def validate(self):
        """
        Validate the model
        """
        try:
            results = self.model.base_model.val(data=str(self.data_yaml_path))
            return results
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise

def main():
    trainer = EnhancedYOLOTrainer(
        data_yaml_path='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
        model_type='yolov8n.yaml',
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