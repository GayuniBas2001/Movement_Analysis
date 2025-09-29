# Movement Analysis System

A real-time computer vision system for exercise form evaluation and repetition counting using MediaPipe pose detection and Intel RealSense depth cameras.

## 🎯 Project Overview

This system provides real-time feedback on exercise form, automatically counts repetitions, and offers comprehensive movement analysis using both 2D and 3D pose estimation techniques. It's specifically designed for bicep curl analysis but can be extended to other exercises.

## 📁 File Structure & Components

### Core Analysis Files

#### `pose_detection.py`
**Purpose:** Main 2D pose detection application with real-time feedback
- **What it does:**
  - Captures video from laptop/webcam camera
  - Uses MediaPipe Pose Landmarker for real-time pose detection
  - Calculates joint angles (elbow, shoulder, hip) in 2D space
  - Counts bicep curl repetitions with form validation
  - Displays side-by-side video feed and feedback panel
- **Key Features:**
  - Real-time rep counting with phase tracking
  - Form evaluation with percentage scoring
  - Visual angle overlays on video feed
  - Keyboard controls (q=quit, r=reset counter)
- **Dependencies:** OpenCV, MediaPipe, NumPy

#### `3D_pose_evaluation.py`
**Purpose:** Enhanced pose analysis with Intel RealSense depth camera integration
- **What it does:**
  - Combines MediaPipe 2D pose detection with RealSense depth sensing
  - Calculates 3D-aware joint angles using depth information
  - Provides three-panel display: Color video + Depth video + Feedback panel
  - Offers configurable depth sensor presets for different environments
  - Advanced movement direction validation during rep transitions
- **Key Features:**
  - True 3D angle calculations with depth awareness
  - Multiple depth sensor configurations (indoor, gym, outdoor, close-range)
  - Real-time depth quality monitoring
  - Movement direction validation to prevent incorrect form
  - Enhanced feedback with 3D analysis metrics
- **Dependencies:** OpenCV, MediaPipe, NumPy, pyrealsense2

### Utility & Analysis Modules

#### `angle_utils.py`
**Purpose:** Mathematical utilities for joint angle calculations
- **What it does:**
  - Provides flexible angle calculation functions
  - Supports both 2D (pixel-based) and 3D (depth-aware) calculations
  - Handles edge cases and coordinate boundary checking
- **Key Functions:**
  - `calculate_angle(a, b, c, use_2d=True)`: Core angle calculation between three points
  - `knee_angle(hip, knee, ankle)`: Specialized knee angle calculation
- **Features:**
  - Robust error handling for zero-length vectors
  - Coordinate clamping to prevent out-of-bounds access
  - Switchable 2D/3D calculation modes

#### `pose_evaluation.py`
**Purpose:** Exercise form evaluation and feedback generation
- **What it does:**
  - Analyzes joint angles to determine exercise form quality
  - Generates detailed feedback messages for form improvement
  - Calculates form scores as percentages
  - Validates movement patterns against ideal ranges
- **Key Functions:**
  - `evaluate_bicep_curl(angles)`: Comprehensive bicep curl form analysis
- **Evaluation Criteria:**
  - Elbow angle range (60°-170° for proper curl)
  - Shoulder stability (minimal movement during curl)
  - Hip alignment (preventing body swing)
  - Overall form scoring algorithm

#### `rep_counter.py`
**Purpose:** Intelligent repetition counting with form validation
- **What it does:**
  - Implements state machine for rep phase tracking
  - Validates movement direction during transitions
  - Prevents counting of partial or incorrect movements
  - Maintains global form state with automatic recovery
- **Key Features:**
  - Two-phase state machine: "down" (ready) ↔ "up" (curled)
  - Movement direction validation using wrist Y-coordinate tracking
  - Form-based rep validation (only counts reps with good form)
  - Stability requirements (must hold positions for multiple frames)
  - Automatic form state recovery to prevent deadlocks
- **States:**
  - `READY_POSITION`: Arm extended, ready to curl
  - `CURLED_POSITION`: Arm curled, ready to extend
  - `MID_REP`: Transition state during movement
  - Form status: `CORRECT` or `INCORRECT` based on movement direction

### Testing & Configuration Files

#### `realsense_test.py`
**Purpose:** RealSense camera testing and configuration utility
- **What it does:**
  - Tests Intel RealSense camera connectivity and functionality
  - Provides interactive depth sensor configuration
  - Displays real-time depth and color streams
  - Offers preset configurations for different environments
- **Features:**
  - Multiple depth presets (default, long-range, bright light, close-up)
  - Real-time depth quality metrics
  - Center distance measurement
  - Calibration mode for distance validation

#### `data_logger.py`
**Purpose:** Data logging and session management
- **What it does:**
  - Logs exercise session data to files
  - Saves camera configurations
  - Records pose landmark data for analysis
  - Manages session folders and file organization
- **Features:**
  - Automatic session folder creation
  - JSON-based configuration saving
  - CSV data export for pose landmarks
  - Session metadata tracking

#### `main.py`
**Purpose:** Entry point and application launcher
- **What it does:**
  - Provides command-line interface for the application
  - Handles application initialization and setup
  - Manages different execution modes (2D vs 3D analysis)
- **Features:**
  - User-friendly application startup
  - Mode selection interface
  - Error handling and graceful shutdown

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
```

### Required Hardware
- **For 2D Analysis:** Any USB webcam or laptop camera
- **For 3D Analysis:** Intel RealSense depth camera (D400 series recommended)

### Running the Applications

#### 2D Pose Analysis
```bash
python pose_detection.py
```
- Uses standard webcam
- Provides real-time rep counting and form feedback
- Lightweight and works on most systems

#### 3D Pose Analysis
```bash
python 3D_pose_evaluation.py
```
- Requires Intel RealSense camera
- Advanced depth-aware analysis
- More accurate angle calculations
- Multiple environment presets

### Controls
- **Q:** Quit application
- **R:** Reset rep counter
- **1-4:** Switch depth presets (3D mode only)

## 📊 Features

### Real-Time Analysis
- ✅ Live pose detection with 33 body landmarks
- ✅ Automatic rep counting with form validation
- ✅ Real-time angle calculations and display
- ✅ Movement direction validation
- ✅ Form scoring with percentage feedback

### 3D Depth Integration
- ✅ Intel RealSense depth camera support
- ✅ True 3D angle calculations
- ✅ Depth quality monitoring
- ✅ Multiple environment presets
- ✅ Depth-aware landmark positioning

### Form Evaluation
- ✅ Comprehensive bicep curl analysis
- ✅ Joint angle validation
- ✅ Movement pattern recognition
- ✅ Detailed feedback messages
- ✅ Form score percentage calculation

### User Interface
- ✅ Multi-panel video display
- ✅ Real-time feedback overlay
- ✅ Phase tracking visualization
- ✅ Angle measurement display
- ✅ Rep counter with form validation

## 🔧 Technical Architecture

### Core Technologies
- **MediaPipe:** Pose detection and landmark extraction
- **OpenCV:** Video processing and display
- **Intel RealSense SDK:** Depth sensing and 3D coordinates
- **NumPy:** Mathematical operations and array processing

### Algorithm Flow
1. **Video Capture:** Real-time video stream acquisition
2. **Pose Detection:** MediaPipe landmark extraction (33 points)
3. **Coordinate Processing:** Convert normalized to pixel coordinates
4. **Depth Integration:** Sample depth values at landmark positions (3D mode)
5. **Angle Calculation:** Compute joint angles using vector mathematics
6. **Form Evaluation:** Analyze angles against ideal ranges
7. **Rep Counting:** State machine with movement validation
8. **Feedback Generation:** Real-time user guidance and scoring
9. **Display:** Multi-panel visualization with overlays

## 📈 Future Enhancements

### Planned Features
- [ ] Additional exercise types (squats, push-ups, etc.)
- [ ] Machine learning-based form classification
- [ ] Historical data analysis and progress tracking
- [ ] Mobile app integration
- [ ] Multi-person analysis support
- [ ] Real-time coaching recommendations

### Technical Improvements
- [ ] True 3D world coordinate calculations
- [ ] Camera calibration for improved accuracy
- [ ] Advanced noise filtering for smoother tracking
- [ ] GPU acceleration for faster processing
- [ ] Cloud-based data synchronization

## 🤝 Contributing

This project is designed for exercise form analysis and can be extended for various fitness applications. The modular architecture makes it easy to add new exercises, evaluation criteria, and analysis features.

## 📄 License

This project is for educational and research purposes. Please ensure proper attribution when using or modifying the code.

---

**Created by:** Movement Analysis Team  
**Last Updated:** September 2025  
**Version:** 2.0 (3D Integration)
