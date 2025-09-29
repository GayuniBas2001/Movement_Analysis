# 3D Movement Analysis - ML Training Pipeline

This directory contains the machine learning training infrastructure for developing more accurate exercise form evaluation models using depth video data and pose landmarks.

## Overview

The ML training pipeline extends the rule-based evaluation system by:

- Collecting synchronized RGB/depth video data with rule-based evaluation labels
- Training neural networks to classify exercise form quality
- Using temporal attention mechanisms for movement pattern recognition
- Supporting multiple exercise types with shared and exercise-specific architectures

## Directory Structure

```
model_training/
├── utils/                      # Utility functions copied from backend
│   ├── pose_evaluation.py      # Rule-based evaluation functions
│   ├── rep_counter.py          # Exercise-specific rep counters
│   ├── angle_utils.py          # 3D angle calculation utilities
│   ├── landmark_utils.py       # 3D landmark extraction
│   └── angle_calculator.py     # Selective angle calculator
├── data_collection/            # Data collection scripts
│   └── collect_training_data.py # RGB/depth video + label collection
├── models/                     # ML model architectures
│   └── movement_analysis_model.py # Neural network models
├── training/                   # Training scripts and utilities
│   └── train_model.py          # Main training script
├── collected_data/             # Collected training data (created during collection)
│   ├── rgb_videos/            # RGB video files
│   ├── depth_videos/          # Depth video files
│   └── labels/                # JSON files with frame-level labels
└── requirements.txt           # Python dependencies
```

## Key Components

### 1. Data Collection (`data_collection/`)

- **collect_training_data.py**: Captures synchronized RGB and depth video while generating rule-based evaluation labels for each frame
- Uses Intel RealSense D435i for true 3D coordinates
- MediaPipe Pose Landmarker (heavy model) for 33-point pose detection
- Real-time form evaluation provides ground truth labels

### 2. Neural Network Models (`models/`)

- **PoseLandmarkEncoder**: Encodes 33 3D landmarks into feature representations
- **TemporalAttention**: Focuses on important movement phases during exercises
- **MovementAnalysisModel**: Complete architecture with LSTM + attention for temporal modeling
- **ExerciseSpecificModel**: Multi-exercise architecture with shared encoding

### 3. Training Pipeline (`training/`)

- **MovementDataset**: PyTorch dataset for loading collected video sequences
- **MovementTrainer**: Training manager with combined classification and regression losses
- Support for multiple exercise types and sequence-based learning

### 4. Utility Functions (`utils/`)

Essential functions copied from the backend system:

- Rule-based evaluation for generating training labels
- 3D coordinate conversion using camera intrinsics
- Exercise-specific angle calculations
- Rep counting logic for movement validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
cd data_collection
python collect_training_data.py
```

This will:

- Start the RealSense camera
- Display real-time pose detection and form evaluation
- Save RGB video, depth video, and frame-level labels
- Default: 60-second collection session for Bicep Curl exercise

### 3. Train ML Model

```bash
cd training
python train_model.py
```

This will:

- Load collected data and create training sequences
- Train a neural network with LSTM + attention architecture
- Save the best model based on validation performance
- Generate training history plots

### 4. Evaluate Model Performance

The trained models output:

- **Form Classification**: 5-class rating (Excellent, Good, Fair, Poor, Very Poor)
- **Form Score Regression**: Continuous score 0-100% matching rule-based system

## Technical Details

### Data Format

Each training sample contains:

- **landmarks_3d**: 33 pose landmarks with (x_pixel, y_pixel, depth_meters) coordinates
- **rule_based_labels**: Ground truth from pose_evaluation.py functions
  - feedback: Text feedback on form issues
  - form_score: Numerical score 0-100
  - is_good_form: Boolean form acceptability
  - calculated_angles: Joint angles used in evaluation

### Model Architecture

1. **Input**: Sequences of 30 frames × 99 features (33 landmarks × 3 coordinates)
2. **Landmark Encoder**: Multi-layer perceptron for feature extraction
3. **LSTM**: Bidirectional LSTM for temporal pattern recognition
4. **Attention**: Temporal attention to focus on key movement phases
5. **Output Heads**:
   - Classification head for 5-class form rating
   - Regression head for continuous form score

### Training Strategy

- **Sequence Length**: 30 frames (~1 second at 30 FPS)
- **Sequence Overlap**: 50% overlap between training sequences
- **Loss Function**: Combined classification + regression loss
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers and early stopping

## Exercise Support

### Currently Implemented

- **Bicep Curl**: Elbow angle tracking with movement direction validation

### Extensible Architecture

The system is designed to easily add new exercises by:

1. Adding evaluation functions to `utils/pose_evaluation.py`
2. Creating exercise-specific rep counters in `utils/rep_counter.py`
3. Updating angle calculation requirements in `utils/angle_calculator.py`

## Integration with Main System

The trained ML models can be integrated back into the main 3D pose evaluation system (`backend/3D_pose_evaluation.py`) to replace or augment rule-based evaluation:

```python
# Load trained model
model = torch.load('movement_model_Bicep_Curl_20241213.pth')

# Use for inference during live evaluation
form_class, form_score = model(landmark_sequence)
```

## Future Enhancements

1. **Multi-Modal Learning**: Incorporate RGB features alongside pose landmarks
2. **Real-Time Inference**: Optimize models for live evaluation performance
3. **Personalization**: Adapt models to individual user movement patterns
4. **Advanced Architectures**: Explore transformer-based temporal modeling
5. **Data Augmentation**: Synthetic pose variations for improved robustness

## Performance Expectations

With sufficient training data (1000+ exercise repetitions per type), the ML models should achieve:

- **Classification Accuracy**: >85% for 5-class form rating
- **Regression Accuracy**: MAE <10 points on 0-100 form score scale
- **Inference Speed**: <50ms per sequence on modern GPUs

The ML approach provides several advantages over rule-based evaluation:

- **Learned Features**: Discovers subtle movement patterns beyond explicit rules
- **Temporal Context**: Better understanding of movement flow and transitions
- **Adaptability**: Learns from diverse users and movement variations
- **Robustness**: Less sensitive to landmark detection noise and occlusions
