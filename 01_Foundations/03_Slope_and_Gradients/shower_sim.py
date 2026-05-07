import time

def adjust_shower():
    target_temp = 40.0
    current_temp = 20.0  # Cold start
    knob_position = 0.0  # 0 to 10 scale
    learning_rate = 0.1  # Smaller nudges for stability
    
    print("--- Smart Shower AI ---")
    print(f"Goal: {target_temp}°C")
    print("-" * 30)

    for i in range(1, 11):
        # 1. Loss: How much are we shivering/burning?
        loss = (current_temp - target_temp) ** 2
        
        # 2. Gradient: In this shower, 1 turn adds 2 degrees
        # (This is the relationship between the knob and the result)
        sensitivity = 2.0 
        
        # 3. The Adjustment
        # If temp is too low, move knob UP. If too high, move knob DOWN.
        direction = 1 if current_temp < target_temp else -1
        
        # How much to nudge? (Loss tells us how urgent it is)
        # In real math, the gradient includes the Loss info
        nudge = learning_rate * direction * (abs(current_temp - target_temp) / sensitivity)
        
        knob_position += nudge
        current_temp = knob_position * 10 # Simple rule: Knob 4 = 40 degrees
        
        print(f"Step {i}: Temp = {current_temp:.1f}°C | Knob = {knob_position:.2f} | Loss = {loss:.1f}")
        
        if loss < 0.1:
            print("\nPerfect! The water is just right. 🚿")
            break
        
        time.sleep(0.3)

if __name__ == "__main__":
    adjust_shower()
