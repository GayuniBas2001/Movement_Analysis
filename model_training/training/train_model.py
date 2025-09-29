"""
Training Script for Movement Analysis ML Models

This script trains neural networks to classify exercise form quality
using collected depth video data and rule-based evaluation labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Add model and utils paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from movement_analysis_model import create_movement_model


class MovementDataset(Dataset):
    """
    Dataset class for loading movement analysis training data.
    """
    def __init__(self, data_dir, sequence_length=30, exercise_type=None):
        """
        Args:
            data_dir: Directory containing collected training data
            sequence_length: Number of frames in each training sequence
            exercise_type: Specific exercise type to filter (None for all)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.exercise_type = exercise_type
        
        # Load all label files
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} training samples")
    
    def _load_samples(self):
        """Load and process all training samples."""
        samples = []
        labels_dir = os.path.join(self.data_dir, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory not found: {labels_dir}")
            return samples
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.json'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    session_data = json.load(f)
                
                metadata = session_data.get('metadata', {})
                frame_labels = session_data.get('frame_labels', [])
                
                # Filter by exercise type if specified
                if (self.exercise_type and 
                    metadata.get('exercise_type') != self.exercise_type):
                    continue
                
                # Create sequences from frame data
                sequences = self._create_sequences(frame_labels, metadata)
                samples.extend(sequences)
                
            except Exception as e:
                print(f"Error loading {label_file}: {e}")
                continue
        
        return samples
    
    def _create_sequences(self, frame_labels, metadata):
        """Create training sequences from frame labels."""
        sequences = []
        
        if len(frame_labels) < self.sequence_length:
            return sequences
        
        # Create overlapping sequences
        step_size = self.sequence_length // 2  # 50% overlap
        
        for start_idx in range(0, len(frame_labels) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            
            sequence_frames = frame_labels[start_idx:end_idx]
            
            # Extract landmarks and labels
            landmarks_sequence = []
            form_scores = []
            is_good_form_flags = []
            
            valid_sequence = True
            
            for frame_data in sequence_frames:
                try:
                    # Extract 3D landmarks (33 landmarks * 3 coordinates = 99 features)
                    landmarks_3d = frame_data['landmarks_3d']
                    if len(landmarks_3d) != 33:
                        valid_sequence = False
                        break
                    
                    # Flatten landmarks to feature vector
                    landmark_features = []
                    for landmark in landmarks_3d:
                        landmark_features.extend(landmark[:3])  # x, y, depth
                    
                    landmarks_sequence.append(landmark_features)
                    
                    # Extract labels
                    rule_labels = frame_data['rule_based_labels']
                    form_scores.append(rule_labels['form_score'])
                    is_good_form_flags.append(rule_labels['is_good_form'])
                    
                except (KeyError, IndexError, TypeError) as e:
                    valid_sequence = False
                    break
            
            if not valid_sequence:
                continue
            
            # Create sequence sample
            sequence_sample = {
                'landmarks': np.array(landmarks_sequence, dtype=np.float32),
                'form_score': np.mean(form_scores),
                'form_class': self._score_to_class(np.mean(form_scores)),
                'good_form_ratio': np.mean(is_good_form_flags),
                'exercise_type': metadata.get('exercise_type', 'Unknown'),
                'session_id': metadata.get('session_id', 'Unknown')
            }
            
            sequences.append(sequence_sample)
        
        return sequences
    
    def _score_to_class(self, form_score):
        """Convert form score to discrete class."""
        if form_score >= 90:
            return 4  # Excellent
        elif form_score >= 80:
            return 3  # Good
        elif form_score >= 70:
            return 2  # Fair
        elif form_score >= 60:
            return 1  # Poor
        else:
            return 0  # Very Poor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        landmarks = torch.FloatTensor(sample['landmarks'])
        form_score = torch.FloatTensor([sample['form_score']])
        form_class = torch.LongTensor([sample['form_class']])
        
        return {
            'landmarks': landmarks,
            'form_score': form_score,
            'form_class': form_class,
            'exercise_type': sample['exercise_type']
        }


class MovementTrainer:
    """
    Training manager for movement analysis models.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in train_loader:
            landmarks = batch['landmarks'].to(self.device)
            form_scores = batch['form_score'].to(self.device).squeeze()
            form_classes = batch['form_class'].to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_classes, pred_scores = self.model(landmarks)
            
            # Calculate losses
            class_loss = self.classification_loss(pred_classes, form_classes)
            score_loss = self.regression_loss(pred_scores.squeeze(), form_scores)
            
            # Combined loss (weighted)
            total_batch_loss = class_loss + 0.1 * score_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(pred_classes.data, 1)
            total_samples += form_classes.size(0)
            correct_predictions += (predicted == form_classes).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                landmarks = batch['landmarks'].to(self.device)
                form_scores = batch['form_score'].to(self.device).squeeze()
                form_classes = batch['form_class'].to(self.device).squeeze()
                
                # Forward pass
                pred_classes, pred_scores = self.model(landmarks)
                
                # Calculate losses
                class_loss = self.classification_loss(pred_classes, form_classes)
                score_loss = self.regression_loss(pred_scores.squeeze(), form_scores)
                total_batch_loss = class_loss + 0.1 * score_loss
                
                total_loss += total_batch_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(pred_classes.data, 1)
                total_samples += form_classes.size(0)
                correct_predictions += (predicted == form_classes).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, save_path='best_model.pth'):
        """Train the model."""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, save_path)
                print(f"  New best model saved!")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            print()
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


def main():
    """Main training script."""
    # Configuration
    data_dir = "../collected_data"
    exercise_type = "Bicep Curl"  # None for all exercises
    sequence_length = 30
    batch_size = 16
    num_epochs = 100
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = MovementDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        exercise_type=exercise_type
    )
    
    if len(full_dataset) == 0:
        print("No training data found! Please run data collection first.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = create_movement_model('standard', num_classes=5)
    
    # Create trainer
    trainer = MovementTrainer(model)
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../models/movement_model_{exercise_type.replace(' ', '_')}_{timestamp}.pth"
    
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, save_path=model_path)
    
    # Plot results
    trainer.plot_training_history(f"../models/training_history_{timestamp}.png")
    
    print(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()