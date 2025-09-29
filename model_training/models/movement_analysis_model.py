"""
3D Movement Analysis Neural Network Models

This module contains ML models for classifying exercise form quality
using depth video data and pose landmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoseLandmarkEncoder(nn.Module):
    """
    Encode 3D pose landmarks into a feature representation.
    """
    def __init__(self, input_dim=99, hidden_dim=256, output_dim=128):
        """
        Args:
            input_dim: Input dimension (33 landmarks * 3 coordinates = 99)
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super(PoseLandmarkEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, landmarks):
        """
        Args:
            landmarks: Tensor of shape (batch_size, sequence_length, 99)
        Returns:
            encoded_features: Tensor of shape (batch_size, sequence_length, output_dim)
        """
        batch_size, seq_len, _ = landmarks.shape
        landmarks_flat = landmarks.view(-1, landmarks.shape[-1])
        encoded = self.encoder(landmarks_flat)
        return encoded.view(batch_size, seq_len, -1)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for focusing on important time steps.
    """
    def __init__(self, input_dim=128, attention_dim=64):
        super(TemporalAttention, self).__init__()
        
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            attended_output: Tensor of shape (batch_size, sequence_length, attention_dim)
        """
        Q = self.query(x)  # (batch_size, seq_len, attention_dim)
        K = self.key(x)    # (batch_size, seq_len, attention_dim)
        V = self.value(x)  # (batch_size, seq_len, attention_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, V)
        
        return attended_output


class MovementAnalysisModel(nn.Module):
    """
    Complete movement analysis model for exercise form classification.
    
    Architecture:
    1. Pose Landmark Encoder: Encodes 3D landmarks into features
    2. LSTM: Captures temporal dependencies in movement
    3. Temporal Attention: Focuses on important movement phases
    4. Classification Head: Outputs form quality predictions
    """
    def __init__(self, 
                 num_landmarks=33, 
                 coordinate_dim=3,
                 encoder_hidden=256,
                 encoder_output=128,
                 lstm_hidden=256,
                 lstm_layers=2,
                 attention_dim=64,
                 num_classes=5):
        """
        Args:
            num_landmarks: Number of pose landmarks (33 for MediaPipe)
            coordinate_dim: Dimension of each coordinate (3 for x,y,depth)
            encoder_hidden: Hidden dimension for landmark encoder
            encoder_output: Output dimension for landmark encoder
            lstm_hidden: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            attention_dim: Attention mechanism dimension
            num_classes: Number of form quality classes (e.g., 5 for 1-5 rating)
        """
        super(MovementAnalysisModel, self).__init__()
        
        input_dim = num_landmarks * coordinate_dim
        
        # Landmark encoder
        self.landmark_encoder = PoseLandmarkEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            output_dim=encoder_output
        )
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=encoder_output,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = TemporalAttention(
            input_dim=lstm_hidden * 2,  # Bidirectional LSTM
            attention_dim=attention_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Additional regression head for form score (0-100)
        self.form_score_regressor = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1, multiply by 100 for percentage
        )
        
    def forward(self, landmarks, return_attention=False):
        """
        Args:
            landmarks: Tensor of shape (batch_size, sequence_length, 99)
            return_attention: Whether to return attention weights
        
        Returns:
            form_class: Class probabilities (batch_size, num_classes)
            form_score: Form score regression (batch_size, 1)
            attention_weights: Optional attention weights
        """
        # Encode landmarks
        encoded_landmarks = self.landmark_encoder(landmarks)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(encoded_landmarks)
        
        # Apply attention
        attended_features = self.attention(lstm_out)
        
        # Global pooling over time dimension
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Classification and regression
        form_class = self.classifier(pooled_features)
        form_score = self.form_score_regressor(pooled_features) * 100  # Scale to 0-100
        
        if return_attention:
            # Return attention weights for visualization
            return form_class, form_score, attended_features
        
        return form_class, form_score


class ExerciseSpecificModel(nn.Module):
    """
    Exercise-specific model that can handle multiple exercise types.
    """
    def __init__(self, 
                 exercise_types=['Bicep Curl'],
                 shared_encoder_dim=128,
                 **model_kwargs):
        super(ExerciseSpecificModel, self).__init__()
        
        self.exercise_types = exercise_types
        self.num_exercises = len(exercise_types)
        
        # Shared landmark encoder
        self.shared_encoder = PoseLandmarkEncoder(
            input_dim=99,  # 33 landmarks * 3 coordinates
            hidden_dim=256,
            output_dim=shared_encoder_dim
        )
        
        # Exercise-specific branches
        self.exercise_branches = nn.ModuleDict()
        for exercise in exercise_types:
            self.exercise_branches[exercise.replace(' ', '_')] = MovementAnalysisModel(
                encoder_output=shared_encoder_dim,
                **model_kwargs
            )
    
    def forward(self, landmarks, exercise_type):
        """
        Args:
            landmarks: Tensor of shape (batch_size, sequence_length, 99)
            exercise_type: String indicating exercise type
        
        Returns:
            form_class: Class probabilities
            form_score: Form score regression
        """
        # Shared encoding
        encoded_landmarks = self.shared_encoder(landmarks)
        
        # Exercise-specific processing
        exercise_key = exercise_type.replace(' ', '_')
        if exercise_key in self.exercise_branches:
            branch_model = self.exercise_branches[exercise_key]
            
            # Use the branch model (skip its encoder since we use shared)
            lstm_out, _ = branch_model.lstm(encoded_landmarks)
            attended_features = branch_model.attention(lstm_out)
            pooled_features = torch.mean(attended_features, dim=1)
            
            form_class = branch_model.classifier(pooled_features)
            form_score = branch_model.form_score_regressor(pooled_features) * 100
            
            return form_class, form_score
        else:
            raise ValueError(f"Exercise type '{exercise_type}' not supported")


# Model factory function
def create_movement_model(model_type='standard', **kwargs):
    """
    Factory function to create different model variants.
    
    Args:
        model_type: Type of model ('standard', 'exercise_specific')
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model
    """
    if model_type == 'standard':
        return MovementAnalysisModel(**kwargs)
    elif model_type == 'exercise_specific':
        return ExerciseSpecificModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_movement_model('standard')
    
    # Mock input: batch_size=2, sequence_length=30, landmarks=99
    test_input = torch.randn(2, 30, 99)
    
    form_class, form_score = model(test_input)
    
    print(f"Model output shapes:")
    print(f"Form class: {form_class.shape}")
    print(f"Form score: {form_score.shape}")
    print(f"Sample form score: {form_score[0].item():.1f}%")