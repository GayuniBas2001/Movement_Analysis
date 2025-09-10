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
        
    def evaluate_rep(self, elbow_angle, form_is_good=True):
        """
        Evaluate if a rep should be counted based on elbow angle and form.
        
        Args:
            elbow_angle: Current elbow angle in degrees
            form_is_good: Boolean indicating if the form is acceptable
            
        Returns:
            tuple: (rep_counted, current_phase, rep_count)
        """
        rep_counted = False
        
        # Only proceed if form is good
        if not form_is_good:
            return rep_counted, self.current_phase, self.rep_count
        
        # State machine for rep phases
        if self.current_phase == "down":
            if elbow_angle <= self.max_elbow_for_up:
                # Moving from down to up position
                self.up_hold_count += 1
                if self.up_hold_count >= self.phase_hold_frames:
                    self.current_phase = "up"
                    self.up_hold_count = 0
                    self.down_hold_count = 0
            else:
                self.up_hold_count = 0
                
        elif self.current_phase == "up":
            if elbow_angle >= self.min_elbow_for_down:
                # Moving from up to down position - COMPLETE REP!
                self.down_hold_count += 1
                if self.down_hold_count >= self.phase_hold_frames:
                    self.current_phase = "down"
                    self.rep_count += 1
                    rep_counted = True
                    self.down_hold_count = 0
                    self.up_hold_count = 0
            else:
                self.down_hold_count = 0
        
        return rep_counted, self.current_phase, self.rep_count
    
    def reset_counter(self):
        """Reset the rep counter"""
        self.rep_count = 0
        self.current_phase = "down"
        self.down_hold_count = 0
        self.up_hold_count = 0
    
    def get_phase_description(self):
        """Get human-readable phase description"""
        if self.current_phase == "down":
            return "Ready Position"
        elif self.current_phase == "up":
            return "Curled Position"
        else:
            return "Transitioning"
