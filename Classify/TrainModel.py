import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
from PIL import Image
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler

# Optional: Import WandB only if used
try:
    import wandb
except ImportError:
    wandb = None  # Make sure this fails gracefully if WandB is not installed

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_val):
        if self.best_loss is None:
            self.best_loss = current_val
        elif current_val > self.best_loss + self.min_delta and self.mode == 'min':
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_val
            self.counter = 0

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler=None,
        device: str = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = scheduler or optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.scaler = GradScaler()
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Setup WandB if enabled in config
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb and wandb:
            self.run = wandb.init(
                project=config['project_name'],
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Save paths
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': running_loss/(batch_idx+1),
                    'acc': 100.*correct/total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Log confusion matrix to WandB if enabled
        if self.use_wandb and wandb:
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Validation Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging to WandB if enabled
            if self.use_wandb and wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Save final model
        self.save_model('final_model.pth')
        if self.use_wandb and wandb:
            self.run.finish()
    
    def save_model(self, filename: str):
        """Save model with config and metrics"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }, save_path)
        if self.use_wandb and wandb:
            wandb.save(save_path)


