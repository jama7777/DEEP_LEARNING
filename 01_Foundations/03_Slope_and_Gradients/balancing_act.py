import time

def simulate_balance():
    # Goal: Get the stick to 90 degrees
    current_angle = 45.0  # It's leaning heavily!
    hand_position = 0.0
    learning_rate = 0.05
    
    print("--- AI BALANCING ACT ---")
    print(f"Initial Angle: {current_angle} degrees (Target: 90)\n")
    print(f"{'Step':<5} | {'Angle':<10} | {'Loss':<10} | {'Slope':<10} | {'Action'}")
    print("-" * 60)

    for i in range(1, 16):
        # 1. Calculate Loss (How bad is the lean?)
        # We use (Angle - 90)^2
        loss = (current_angle - 90.0) ** 2
        
        # 2. Calculate Slope (Sensitivity)
        # In this simple math, the slope of (x-90)^2 is 2*(x-90)
        slope = 2 * (current_angle - 90.0)
        
        # 3. The Gradient Update
        # We nudge the angle based on the slope
        # Note: In real life, hand moves the angle. 
        # Here we directly adjust the angle for simplicity.
        adjustment = slope * learning_rate
        current_angle -= adjustment
        
        action = "Moving Right" if adjustment < 0 else "Moving Left"
        if loss < 0.1: action = "Balanced! 💃"

        print(f"{i:<5} | {current_angle:<10.2f} | {loss:<10.2f} | {slope:<10.2f} | {action}")
        
        if loss < 0.01:
            break
        time.sleep(0.1)

    print("\nDeep Summary:")
    print("1. Loss: Told the AI how 'painful' the lean was.")
    print("2. Slope: Told the AI 'If you move this way, the pain goes up'.")
    print("3. Gradient: The AI moved the OPPOSITE of the slope to find peace (90.0).")

if __name__ == "__main__":
    simulate_balance()
