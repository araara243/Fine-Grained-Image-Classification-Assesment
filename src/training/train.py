import datetime
import argparse
import os
import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import time
import copy
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Determine project root for absolute paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.loader import Flowers102Dataset, get_transforms
from utils.common import set_seed, setup_logger, save_checkpoint

def setup_datasets_and_loaders(data_dir, batch_size=32, dry_run=False, seed=42, num_workers=4):
    """
    Create datasets, dataloaders, and return sizes for Flowers102 training.
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for DataLoaders
        dry_run (bool): If True, use subset of data for testing
        seed (int): Random seed for reproducibility
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (image_datasets, dataloaders, dataset_sizes)
    """
    # Create datasets
    image_datasets = {
        x: Flowers102Dataset(root=data_dir, split=x, transform=get_transforms(x), download=True, seed=seed)
        for x in ['train', 'val']
    }
    
    # Handle dry run mode
    if dry_run:
        image_datasets = {x: torch.utils.data.Subset(image_datasets[x], range(10)) for x in ['train', 'val']}
    
    # Create dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val']
    }
    
    # Calculate dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    return image_datasets, dataloaders, dataset_sizes

def save_model_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best=False, checkpoint_name='checkpoint.pth.tar', checkpoint_dir=None):
    """
    Unified checkpoint saving function
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer  
        epoch: Current epoch number
        val_loss: Validation loss
        val_acc: Validation accuracy
        is_best: Whether this is the best model
        checkpoint_name: Name of the checkpoint file
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'optimizer': optimizer.state_dict()
    }
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(project_root, 'src', 'model')
    save_checkpoint(checkpoint, is_best=is_best, checkpoint_dir=checkpoint_dir, filename=checkpoint_name)
    
    if is_best:
        # Also save as best model
        save_checkpoint(checkpoint, is_best=True, checkpoint_dir=checkpoint_dir, filename='model_best.pth.tar')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', logger=None, writer=None, patience=7):
    since = time.time()

    val_acc_history = []
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(project_root, 'src', 'model', 'early_stop.pth'))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        early_stopping.epoch = epoch
        if logger:
            logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if logger:
                logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # TensorBoard logging
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
                # Check early stopping
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    if logger:
                        logger.info("Early stopping")
                    # Load best model weights so far to return
                    model.load_state_dict(best_model_wts) 
                    return model, val_acc_history

        if logger:
            logger.info('')

    time_elapsed = time.time() - since
    if logger:
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    parser = argparse.ArgumentParser(description='Train Flowers102 Classifier')
    parser.add_argument('--data_dir', type=str, default='../src/data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
    parser.add_argument('--dry_run', action='store_true', help='Run a single epoch for testing')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Determine project root (relative to this script: src/training/train.py -> ../../)
    logs_dir = os.path.join(project_root, 'logs')

    # Setup logging
    logger = setup_logger('train_logger', os.path.join(logs_dir, 'train.log'))
    logger.info(f"Starting training with args: {args}")
    
    # Setup TensorBoard
    # Terminal run format: logs\tensorboard\run<datemonthyear-time>\train\
    timestamp = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
    log_dir_tb = os.path.join(logs_dir, 'tensorboard', f'run{timestamp}', 'train')
    writer = SummaryWriter(log_dir_tb)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Datasets and dataloaders
    image_datasets, dataloaders, dataset_sizes = setup_datasets_and_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        seed=args.seed
    )
    logger.info(f"Dataset sizes: {dataset_sizes}")

    # Model
    # Using ResNet18 pretrained on ImageNet
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    # Flowers102 has 102 classes
    model_ft.fc = torch_nn.Linear(num_ftrs, 102)

    model_ft = model_ft.to(device)

    criterion = torch_nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

    # Dry run check
    epochs = 1 if args.dry_run else args.epochs

    # Train
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=epochs, device=device, logger=logger, writer=writer, patience=args.patience)

    # Save best model
    best_acc = max(hist) if hist else 0.0
    save_model_checkpoint(
        model=model_ft,
        optimizer=optimizer_ft,
        epoch=epochs,
        val_loss=0.0,  # We don't track final loss in main()
        val_acc=best_acc,
        is_best=True,
        checkpoint_name='checkpoint_final.pth.tar',
        checkpoint_dir=os.path.join(project_root, 'src', 'model')
    )
    
    writer.close()

if __name__ == '__main__':
    main()
