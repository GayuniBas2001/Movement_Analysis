from pose_evaluation import (
    evaluate_bicep_curl, 
    evaluate_arm_raise, 
    evaluate_trunk_rotation, 
    evaluate_sit_to_stand, 
    evaluate_squat, 
    evaluate_heel_rise
)

def select_exercise():
    """Display exercise selection menu and return selected exercise"""
    exercises = {
        '1': ('Bicep Curl', evaluate_bicep_curl),
        '2': ('Arm Raise', evaluate_arm_raise),
        '3': ('Trunk Rotation', evaluate_trunk_rotation),
        '4': ('Sit to Stand', evaluate_sit_to_stand),
        '5': ('Squat', evaluate_squat),
        '6': ('Heel Rise', evaluate_heel_rise)
    }
    
    print("="*50)
    print("MOVEMENT ANALYSIS - EXERCISE SELECTION")
    print("="*50)
    print("Select the exercise you want to perform:")
    print()
    for key, (name, _) in exercises.items():
        print(f"{key}. {name}")
    print()
    
    while True:
        choice = input("Enter your choice (1-6): ").strip()
        if choice in exercises:
            selected_name, selected_func = exercises[choice]
            print(f"\nSelected Exercise: {selected_name}")
            print("="*50)
            return selected_name, selected_func
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")