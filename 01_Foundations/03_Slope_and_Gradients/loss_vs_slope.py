import time

def compare_methods():
    target = 90.0
    
    # --- AI 1: ONLY LOSS (The Blind AI) ---
    angle_loss_only = 110.0 # It's leaning RIGHT
    print("--- AI 1: USING ONLY LOSS (No Direction) ---")
    print("This AI thinks: 'My Loss is 400, so I will just move 400!'")
    
    for i in range(1, 4):
        loss = (angle_loss_only - target) ** 2
        # It doesn't know direction, so it just ADDS the loss
        angle_loss_only += loss 
        print(f"Step {i}: Angle = {angle_loss_only:.1f} | Loss = {loss:.1f} (OH NO!)")
    
    print("\n" + "="*50 + "\n")

    # --- AI 2: USING SLOPE (The Compass AI) ---
    angle_slope = 110.0 # Same starting point
    print("--- AI 2: USING SLOPE (Has Direction) ---")
    print("This AI thinks: 'My Slope is Positive, so I must move LEFT!'")
    
    for i in range(1, 4):
        loss = (angle_slope - target) ** 2
        slope = 2 * (angle_slope - target) # The Gradient
        # It uses the slope to move the CORRECT way
        angle_slope -= (0.1 * slope) 
        print(f"Step {i}: Angle = {angle_slope:.1f} | Loss = {loss:.1f} (Success!)")

if __name__ == "__main__":
    compare_methods()
