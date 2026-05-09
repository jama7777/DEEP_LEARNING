import numpy as np

def step_by_step_xray():
    print("🚶 THE WORD TRAVELER: STEP-BY-STEP X-RAY")
    print("=" * 65)

    # 1. INITIAL POSITION
    # The word 'Sun' starts at (0, 0)
    sun_vector = np.array([0.0, 0.0])
    print(f"START: Sun is at {sun_vector}")
    
    lr = 0.1 # Learning Rate (Small steps)

    # 2. STEP 1: 'The Sun is Bright' (Pulls Sun to the Right [10, 0])
    target_1 = np.array([10.0, 0.0])
    error_1 = target_1 - sun_vector
    sun_vector += lr * error_1
    print(f"STEP 1 (After 'Bright'): Sun moved to {sun_vector}")

    # 3. STEP 2: 'The Sun is Hot' (Pulls Sun UP [0, 10])
    target_2 = np.array([0.0, 10.0])
    error_2 = target_2 - sun_vector
    sun_vector += lr * error_2
    print(f"STEP 2 (After 'Hot'):    Sun moved to {sun_vector}")

    print("\n" + "-" * 65)
    print("💡 THE DEEP TRUTH:")
    print(f"1. Final Position: {sun_vector}")
    print("2. Did we lose 'Bright'? No! The X-coordinate is still 1.0 (Positive).")
    print("3. Did we lose 'Hot'? No! The Y-coordinate is now 0.9 (Positive).")
    print("\nRESULT: The Sun vector now 'Remembers' both facts.")
    print("It is a point that is both to the Right AND Up.")
    print("In 128D, we just have 128 of these 'Facts' being balanced at once!")

if __name__ == "__main__":
    step_by_step_xray()
