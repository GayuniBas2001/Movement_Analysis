class BicepCurlCounter:
    def __init__(self):
        self.rep_count = 0
        self.current_phase = "down"  # "down", "up", "transition"
        self.min_elbow_for_down = 140  # Elbow angle when arm is extended (down position)
        self.max_elbow_for_up = 60     # Elbow angle when arm is curled (up position)
        self.form_threshold = 0.8      # Minimum form score to count rep (80%)
        self.phase_hold_frames = 5     # Frames to hold position before counting
        self.down_hold_count = 0
        self.up_hold_count = 0
        self.transition_state = None   # Global transition state
        self.is_form_good_global = True  # Global form state
        self.previous_wrist_y = None   # Track wrist Y coordinate for direction detection
        self.movement_direction_buffer = []  # Buffer to track consistent movement
        self.form_violation_frames = 0  # Count frames since form violation
        self.max_form_violation_frames = 30  # Reset form after 30 frames (~1 second)
        
    def evaluate_rep(self, elbow_angle, wrist_y_coord, form_is_good=True):
        """
        Evaluate if a rep should be counted based on elbow angle, movement direction, and form.
        
        Args:
            elbow_angle: Current elbow angle in degrees
            wrist_y_coord: Y coordinate of wrist for movement direction tracking
            form_is_good: Boolean indicating if the form is acceptable
            
        Returns:
            tuple: (rep_counted, current_phase, rep_count, transition_state, is_form_good_global)
        """
        rep_counted = False
        
        # Track movement direction
        movement_direction = self._track_movement_direction(wrist_y_coord)
        
        # Validate movement direction during transitions
        self._validate_movement_direction(movement_direction)
        
        # Handle form violation timeout (reset after 1 second of bad form)
        if not self.is_form_good_global:
            self.form_violation_frames += 1
            if self.form_violation_frames >= self.max_form_violation_frames:
                print("🔄 Form reset after timeout - try again")
                self.is_form_good_global = True
                self.form_violation_frames = 0
        else:
            self.form_violation_frames = 0
        
        # Only proceed if form is good
        if not form_is_good or not self.is_form_good_global:
            return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
        
        # State machine for rep phases
        if self.current_phase == "down":
            if elbow_angle <= self.max_elbow_for_up:
                # Moving from down to up position
                self.transition_state = "MID_REP"
                self.up_hold_count += 1
                if self.up_hold_count >= self.phase_hold_frames:
                    self.current_phase = "up"
                    self.transition_state = None
                    self.up_hold_count = 0
                    self.down_hold_count = 0
                    self.is_form_good_global = True  # Reset form state after successful transition
            else:
                self.up_hold_count = 0
                self.transition_state = None
                
        elif self.current_phase == "up":
            if elbow_angle >= self.min_elbow_for_down:
                # Moving from up to down position - COMPLETE REP!
                self.transition_state = "MID_REP"
                self.down_hold_count += 1
                if self.down_hold_count >= self.phase_hold_frames:
                    self.current_phase = "down"
                    self.transition_state = None
                    self.rep_count += 1
                    rep_counted = True
                    self.down_hold_count = 0
                    self.up_hold_count = 0
                    self.is_form_good_global = True  # Reset form state after successful rep
            else:
                self.down_hold_count = 0
                self.transition_state = None
        
        return rep_counted, self.current_phase, self.rep_count, self.transition_state, self.is_form_good_global
    
    def _track_movement_direction(self, wrist_y_coord):
        """Track the movement direction of the wrist"""
        if self.previous_wrist_y is None:
            self.previous_wrist_y = wrist_y_coord
            return None
        
        # Calculate movement direction (negative = moving up, positive = moving down)
        movement = wrist_y_coord - self.previous_wrist_y
        self.previous_wrist_y = wrist_y_coord
        
        # Add to buffer and keep only last 3 frames for stability
        self.movement_direction_buffer.append(movement)
        if len(self.movement_direction_buffer) > 3:
            self.movement_direction_buffer.pop(0)
        
        # Return average movement direction
        if len(self.movement_direction_buffer) >= 2:
            avg_movement = sum(self.movement_direction_buffer) / len(self.movement_direction_buffer)
            # Debug: print movement info during transitions
            if self.transition_state == "MID_REP" and abs(avg_movement) > 5:
                print(f"🔍 Movement: {avg_movement:.1f} pixels, Phase: {self.current_phase}")
            return avg_movement
        return None
    
    def _validate_movement_direction(self, movement_direction):
        """Validate if movement direction matches expected transition"""
        if movement_direction is None or self.transition_state != "MID_REP":
            return
        
        # Use larger threshold to avoid false negatives from camera noise
        movement_threshold = 10  # pixels
        
        # Define expected movement directions for each transition
        if self.current_phase == "down" and self.up_hold_count > 0:
            # Transitioning from down to up: wrist should move up (decreasing Y in camera coords)
            if movement_direction > movement_threshold:  # Moving down when should move up
                print(f"❌ Wrong direction detected: moving down ({movement_direction:.1f}) during up transition")
                self.is_form_good_global = False
                
        elif self.current_phase == "up" and self.down_hold_count > 0:
            # Transitioning from up to down: wrist should move down (increasing Y in camera coords)
            if movement_direction < -movement_threshold:  # Moving up when should move down
                print(f"❌ Wrong direction detected: moving up ({movement_direction:.1f}) during down transition")
                self.is_form_good_global = False

    def reset_counter(self):
        """Reset the rep counter"""
        self.rep_count = 0
        self.current_phase = "down"
        self.down_hold_count = 0
        self.up_hold_count = 0
        self.transition_state = None
        self.is_form_good_global = True
        self.previous_wrist_y = None
        self.movement_direction_buffer = []
        self.form_violation_frames = 0
    
    def get_phase_description(self):
        """Get human-readable phase description"""
        if self.transition_state == "MID_REP":
            return "Mid-Rep Transition"
        elif self.current_phase == "down":
            return "Ready Position"
        elif self.current_phase == "up":
            return "Curled Position"
        else:
            return "Transitioning"
    
    def get_form_status(self):
        """Get current form status"""
        return "CORRECT" if self.is_form_good_global else "INCORRECT"
