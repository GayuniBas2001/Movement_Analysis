import numpy as np

class ExerciseCounter:
    def __init__(self):
        self.rep_count = 0
        self.current_phase = "down"
        self.phase_hold_frames = 5
        self.down_hold_count = 0
        self.up_hold_count = 0
        self.transition_state = None
        self.is_form_good_global = True
        self.previous_track_y = None
        self.movement_direction_buffer = []
        self.form_violation_frames = 0
        self.max_form_violation_frames = 30

    def evaluate_rep(self, landmarks, calculated_angles, form_is_good=True):
        """Generic rep evaluation (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")

    def _track_movement_direction(self, track_y_coord):
        """Track movement direction using Y coordinate of a landmark."""
        if self.previous_track_y is None:
            self.previous_track_y = track_y_coord
            return None

        movement = track_y_coord - self.previous_track_y
        self.previous_track_y = track_y_coord

        self.movement_direction_buffer.append(movement)
        if len(self.movement_direction_buffer) > 3:
            self.movement_direction_buffer.pop(0)

        if len(self.movement_direction_buffer) >= 2:
            return sum(self.movement_direction_buffer) / len(self.movement_direction_buffer)
        return None

    def _validate_movement_direction(self, movement_direction):
        """Validate if movement direction matches expected transition."""
        if movement_direction is None or self.transition_state != "MID_REP":
            return

        movement_threshold = 10  # pixels

        if self.current_phase == "down" and self.up_hold_count > 0:
            if movement_direction > movement_threshold:
                print("❌ Wrong direction detected during up transition")
                self.is_form_good_global = False
        elif self.current_phase == "up" and self.down_hold_count > 0:
            if movement_direction < -movement_threshold:
                print("❌ Wrong direction detected during down transition")
                self.is_form_good_global = False

    def reset_counter(self):
        """Reset the rep counter."""
        self.rep_count = 0
        self.current_phase = "down"
        self.down_hold_count = 0
        self.up_hold_count = 0
        self.transition_state = None
        self.is_form_good_global = True
        self.previous_track_y = None
        self.movement_direction_buffer = []
        self.form_violation_frames = 0

    def get_phase_description(self):
        if self.transition_state == "MID_REP":
            return "Mid-Rep Transition"
        elif self.current_phase == "down":
            return "Ready Position"
        elif self.current_phase == "up":
            return "Raised Position"
        else:
            return "Transitioning"

    def get_form_status(self):
        return "CORRECT" if self.is_form_good_global else "INCORRECT"


class BicepCurlCounter(ExerciseCounter):
    def __init__(self):
        super().__init__()
        self.min_elbow_for_down = 140
        self.max_elbow_for_up = 60

    def evaluate_rep(self, landmarks, calculated_angles, form_is_good=True):
        rep_counted = False
        elbow_angle = calculated_angles.get("elbow_angle")
        
        # Check if required angle data is available
        if elbow_angle is None:
            print(f"⚠️ Missing elbow angle data - cannot evaluate rep")
            print(f"Available angles: {list(calculated_angles.keys())}")
            return False, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
            
        wrist_y_coord = landmarks[16][1]  # track wrist Y

        # Track + validate movement
        movement_direction = self._track_movement_direction(wrist_y_coord)
        self._validate_movement_direction(movement_direction)

        # Form violation reset
        if not self.is_form_good_global:
            self.form_violation_frames += 1
            if self.form_violation_frames >= self.max_form_violation_frames:
                print("🔄 Form reset after timeout - try again")
                self.is_form_good_global = True
                self.form_violation_frames = 0
        else:
            self.form_violation_frames = 0

        if not form_is_good or not self.is_form_good_global:
            return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global

        # State machine
        if self.current_phase == "down":
            if elbow_angle <= self.max_elbow_for_up:
                self.transition_state = "MID_REP"
                self.up_hold_count += 1
                if self.up_hold_count >= self.phase_hold_frames:
                    self.current_phase = "up"
                    self.transition_state = None
                    self.up_hold_count = 0
                    self.down_hold_count = 0
        elif self.current_phase == "up":
            if elbow_angle >= self.min_elbow_for_down:
                self.transition_state = "MID_REP"
                self.down_hold_count += 1
                if self.down_hold_count >= self.phase_hold_frames:
                    self.current_phase = "down"
                    self.transition_state = None
                    self.rep_count += 1
                    rep_counted = True
                    self.down_hold_count = 0
                    self.up_hold_count = 0

        return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global


class ArmRaiseCounter(ExerciseCounter):
    def __init__(self):
        super().__init__()
        self.min_shoulder_for_down = 40   # Arms down position (below good range)
        self.max_shoulder_for_up = 120    # Arms raised position (within good range)
    
    def get_phase_description(self):
        """Override to provide arm-raise specific descriptions"""
        if self.transition_state == "MID_REP":
            return "Mid-Rep Transition"
        elif self.current_phase == "down":
            return "Arms Down"
        elif self.current_phase == "up":
            return "Arms Raised"
        else:
            return "Transitioning"

    def evaluate_rep(self, landmarks, calculated_angles, form_is_good=True):
        rep_counted = False
        shoulder_angle = calculated_angles.get("shoulder_angle_r")  # Right shoulder angle from arm raise
        
        # Check if required angle data is available
        if shoulder_angle is None:
            print(f"⚠️ Missing shoulder_angle_r data - cannot evaluate rep")
            print(f"Available angles: {list(calculated_angles.keys())}")
            return False, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
            
        wrist_y_coord = landmarks[16][1]

        # Track + validate movement
        movement_direction = self._track_movement_direction(wrist_y_coord)
        self._validate_movement_direction(movement_direction)

        # Form violation reset
        if not self.is_form_good_global:
            self.form_violation_frames += 1
            if self.form_violation_frames >= self.max_form_violation_frames:
                print("🔄 Form reset after timeout - try again")
                self.is_form_good_global = True
                self.form_violation_frames = 0
        else:
            self.form_violation_frames = 0

        if not form_is_good or not self.is_form_good_global:
            return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global

        # State machine
        if self.current_phase == "down":
            if shoulder_angle >= self.max_shoulder_for_up:
                self.transition_state = "MID_REP"
                self.up_hold_count += 1
                print(f"🔄 Arm raise up transition: {shoulder_angle:.1f}° (hold {self.up_hold_count}/{self.phase_hold_frames})")
                if self.up_hold_count >= self.phase_hold_frames:
                    self.current_phase = "up"
                    self.transition_state = None
                    self.up_hold_count = 0
                    self.down_hold_count = 0
                    print("✅ Arms raised position achieved!")
        elif self.current_phase == "up":
            if shoulder_angle <= self.min_shoulder_for_down:
                self.transition_state = "MID_REP"
                self.down_hold_count += 1
                print(f"🔄 Arm raise down transition: {shoulder_angle:.1f}° (hold {self.down_hold_count}/{self.phase_hold_frames})")
                if self.down_hold_count >= self.phase_hold_frames:
                    self.current_phase = "down"
                    self.transition_state = None
                    self.rep_count += 1
                    rep_counted = True
                    self.down_hold_count = 0
                    self.up_hold_count = 0
                    print("✅ Arms lowered - REP COMPLETED!")

        return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global

# class BicepCurlCounter:
#     def __init__(self):
#         self.rep_count = 0
#         self.current_phase = "down"  # "down", "up", "transition"
#         self.min_elbow_for_down = 140  # Elbow angle when arm is extended (down position)
#         self.max_elbow_for_up = 60     # Elbow angle when arm is curled (up position)
#         self.form_threshold = 0.8      # Minimum form score to count rep (80%)
#         self.phase_hold_frames = 5     # Frames to hold position before counting
#         self.down_hold_count = 0
#         self.up_hold_count = 0
#         self.transition_state = None   # Global transition state
#         self.is_form_good_global = True  # Global form state
#         self.previous_wrist_y = None   # Track wrist Y coordinate for direction detection
#         self.movement_direction_buffer = []  # Buffer to track consistent movement
#         self.form_violation_frames = 0  # Count frames since form violation
#         self.max_form_violation_frames = 30  # Reset form after 30 frames (~1 second)
        
#     def evaluate_rep(self, landmarks, calculated_angles, form_is_good=True):
#         """
#         Evaluate if a rep should be counted based on elbow angle, movement direction, and form.
        
#         Args:
#             elbow_angle: Current elbow angle in degrees
#             wrist_y_coord: Y coordinate of wrist for movement direction tracking
#             form_is_good: Boolean indicating if the form is acceptable
            
#         Returns:
#             tuple: (rep_counted, current_phase, rep_count, transition_state, is_form_good_global)
#         """
#         rep_counted = False

#         # elbow_angle = evaluation_parameters.get("elbow_angle")
#         # wrist_y_coord = evaluation_parameters.get("wrist_y_coord")
#         elbow_angle = calculated_angles.get("elbow_angle")
#         wrist_y_coord = landmarks[16][1]  # Assuming landmarks is a dict with body part names as keys
        
#         # Track movement direction
#         movement_direction = self._track_movement_direction(wrist_y_coord)
        
#         # Validate movement direction during transitions
#         self._validate_movement_direction(movement_direction)
        
#         # Handle form violation timeout (reset after 1 second of bad form)
#         if not self.is_form_good_global:
#             self.form_violation_frames += 1
#             if self.form_violation_frames >= self.max_form_violation_frames:
#                 print("🔄 Form reset after timeout - try again")
#                 self.is_form_good_global = True
#                 self.form_violation_frames = 0
#         else:
#             self.form_violation_frames = 0
        
#         # Only proceed if form is good
#         if not form_is_good or not self.is_form_good_global:
#             return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
        
#         # State machine for rep phases
#         if self.current_phase == "down":
#             if elbow_angle <= self.max_elbow_for_up:
#                 # Moving from down to up position
#                 self.transition_state = "MID_REP"
#                 self.up_hold_count += 1
#                 if self.up_hold_count >= self.phase_hold_frames:
#                     self.current_phase = "up"
#                     self.transition_state = None
#                     self.up_hold_count = 0
#                     self.down_hold_count = 0
#                     self.is_form_good_global = True  # Reset form state after successful transition
#             else:
#                 self.up_hold_count = 0
#                 self.transition_state = None
                
#         elif self.current_phase == "up":
#             if elbow_angle >= self.min_elbow_for_down:
#                 # Moving from up to down position - COMPLETE REP!
#                 self.transition_state = "MID_REP"
#                 self.down_hold_count += 1
#                 if self.down_hold_count >= self.phase_hold_frames:
#                     self.current_phase = "down"
#                     self.transition_state = None
#                     self.rep_count += 1
#                     rep_counted = True
#                     self.down_hold_count = 0
#                     self.up_hold_count = 0
#                     self.is_form_good_global = True  # Reset form state after successful rep
#             else:
#                 self.down_hold_count = 0
#                 self.transition_state = None
        
#         return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
    
#     def _track_movement_direction(self, wrist_y_coord):
#         """Track the movement direction of the wrist"""
#         if self.previous_wrist_y is None:
#             self.previous_wrist_y = wrist_y_coord
#             return None
        
#         # Calculate movement direction (negative = moving up, positive = moving down)
#         movement = wrist_y_coord - self.previous_wrist_y
#         self.previous_wrist_y = wrist_y_coord
        
#         # Add to buffer and keep only last 3 frames for stability
#         self.movement_direction_buffer.append(movement)
#         if len(self.movement_direction_buffer) > 3:
#             self.movement_direction_buffer.pop(0)
        
#         # Return average movement direction
#         if len(self.movement_direction_buffer) >= 2:
#             avg_movement = sum(self.movement_direction_buffer) / len(self.movement_direction_buffer)
#             # Debug: print movement info during transitions
#             if self.transition_state == "MID_REP" and abs(avg_movement) > 5:
#                 print(f"🔍 Movement: {avg_movement:.1f} pixels, Phase: {self.current_phase}")
#             return avg_movement
#         return None
    
#     def _validate_movement_direction(self, movement_direction):
#         """Validate if movement direction matches expected transition"""
#         if movement_direction is None or self.transition_state != "MID_REP":
#             return
        
#         # Use larger threshold to avoid false negatives from camera noise
#         movement_threshold = 10  # pixels
        
#         # Define expected movement directions for each transition
#         if self.current_phase == "down" and self.up_hold_count > 0:
#             # Transitioning from down to up: wrist should move up (decreasing Y in camera coords)
#             if movement_direction > movement_threshold:  # Moving down when should move up
#                 print(f"❌ Wrong direction detected: moving down ({movement_direction:.1f}) during up transition")
#                 self.is_form_good_global = False
                
#         elif self.current_phase == "up" and self.down_hold_count > 0:
#             # Transitioning from up to down: wrist should move down (increasing Y in camera coords)
#             if movement_direction < -movement_threshold:  # Moving up when should move down
#                 print(f"❌ Wrong direction detected: moving up ({movement_direction:.1f}) during down transition")
#                 self.is_form_good_global = False

#     def reset_counter(self):
#         """Reset the rep counter"""
#         self.rep_count = 0
#         self.current_phase = "down"
#         self.down_hold_count = 0
#         self.up_hold_count = 0
#         self.transition_state = None
#         self.is_form_good_global = True
#         self.previous_wrist_y = None
#         self.movement_direction_buffer = []
#         self.form_violation_frames = 0
    
#     def get_phase_description(self):
#         """Get human-readable phase description"""
#         if self.transition_state == "MID_REP":
#             return "Mid-Rep Transition"
#         elif self.current_phase == "down":
#             return "Ready Position"
#         elif self.current_phase == "up":
#             return "Curled Position"
#         else:
#             return "Transitioning"
    
#     def get_form_status(self):
#         """Get current form status"""
#         return "CORRECT" if self.is_form_good_global else "INCORRECT"
