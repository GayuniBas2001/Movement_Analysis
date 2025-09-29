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


